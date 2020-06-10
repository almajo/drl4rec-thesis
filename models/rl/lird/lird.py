import numpy as np
import torch
import torch.nn.functional as F

from models.DDPG import DDPG
from models.modules import FeedforwardNetwork
from util.KNN import state_action_concat
from util.exploration import Epsilon_Greedy_Exploration


class LIRD(DDPG):
    agent_name = "LIRD"

    def __init__(self, config):
        super().__init__(config)

    def init_noise(self):
        return Epsilon_Greedy_Exploration(self.config)

    def get_actor(self):
        return FeedforwardNetwork(input_dim=self.get_state_size(),
                                  hidden_dims=self.hyperparameters["Actor"]["linear_hidden_units"],
                                  output_dim=self.embedding.embedding_dim * self.k,
                                  output_fn=self.hyperparameters["Actor"]["final_layer_activation"]).to(self.device)

    def get_critic(self):
        return FeedforwardNetwork(input_dim=self.get_state_size() + self.embedding.embedding_dim * self.k,
                                  hidden_dims=self.hyperparameters["Critic"]["linear_hidden_units"],
                                  output_dim=1).to(self.device)

    def get_batch_rl_actions(self):
        targets = super().get_batch_rl_actions().cpu()
        random_fills = torch.randint(1, self.environment.action_space.n, (targets.shape[0], self.metrics_k - 1))
        td_list = torch.cat([targets.unsqueeze(-1), random_fills], dim=-1)
        return td_list

    def state_to_action(self, state, eval_ep, top_k):
        a = self.actor_local(state)
        a = a.view(a.size(0), top_k, -1)
        a = self.proto_to_real_action(a)
        return a

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, self.info = self.environment.step(action)
        self.total_episode_score_so_far += self.reward.mean()
        if self.hyperparameters["clip_rewards"]:
            self.reward = np.maximum(np.minimum(self.reward, 1.0), -1.0)

    def proto_to_real_action(self, action):
        weight = self.embedding(torch.arange(self.environment.action_space.n, device=self.device)).t().detach()
        actions = []
        m = torch.zeros(action.size(0), weight.size(-1), device=self.device)
        m[:, 0] = -1e15
        for a in range(action.size(1)):
            scores = torch.matmul(action[:, a], weight)
            if actions:
                mask = actions[-1]
                m.scatter_(1, mask, -1e15)
            scores = scores + m
            max_actions = scores.argmax(dim=-1, keepdim=True)
            actions.append(max_actions)
        max_actions = torch.cat(actions, dim=1)
        return max_actions

    def get_q_value(self, net, state, action):
        action = self.embedding(action)
        action = action.view(action.size(0), -1)
        state_action = torch.cat((state, action), 1)
        return net(state_action)

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        actions_next = self.state_to_action(next_states, eval_ep=True, top_k=self.k)
        if isinstance(actions_next, np.ndarray):
            actions_next = torch.as_tensor(actions_next, device=self.device)
        critic_targets_next = self.get_q_value(self.critic_target, next_states, actions_next)
        return critic_targets_next

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.get_q_value(self.critic_local, states, actions)
        self.log_scalar("debug/expected_q_values", critic_expected.mean())
        return critic_expected

    def actor_learn(self, states, targets):
        """Runs a learning iteration for the actor"""
        batch_size = states.size(0)
        item_weights_orig = self.actor_local(states)
        item_weights = item_weights_orig.view(batch_size, self.k, -1)
        if targets is not None:
            t = self.embedding(targets).detach()
            first_item_preds = item_weights[:, 0]
            mse_loss = F.mse_loss(first_item_preds, t)
            self.log_scalar("loss/mse_loss", mse_loss, global_step=self.episode_number)
            self.take_optimisation_step([self.state_optimizer, self.actor_optimizer], self.actor_local, mse_loss,
                                        self.hyperparameters["Actor"]["gradient_clipping_norm"], retain_graph=True)
            self.soft_update_of_target_network(self.actor_local, self.actor_target,
                                               self.hyperparameters["Actor"]["tau"])

        action = self.proto_to_real_action(item_weights)
        action = self.embedding(action)

        action = torch.flatten(action, 1)
        if not action.requires_grad:
            action.requires_grad_(True)
        state_action = torch.cat((states, action), 1)
        loss = - self.critic_local(state_action)

        loss = loss.mean()

        # Delta_a Q(s,a)
        action_grads = torch.autograd.grad(loss, action, only_inputs=True)

        self.actor_optimizer.zero_grad()
        self.state_optimizer.zero_grad()
        # Delta_thetha pi(s | theta) * Q(s,a)
        item_weights_orig.backward(action_grads)
        self.actor_optimizer.step()
        self.state_optimizer.step()

        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute the ranking for the true item to be set on the first position and maximize it
        :param state:
        :param targets:
        :return:
        """
        item_weights_orig = self.actor_local(state)
        item_weights = item_weights_orig.view(state.size(0), self.k, -1)
        first_list_positions = item_weights[:, 0]

        # actor loss
        target_actions = self.embedding(targets).detach()
        actor_loss = F.mse_loss(first_list_positions, target_actions, reduction="none").sum(dim=1)

        # critic loss
        sample_frac = 0.2
        sample_size = int(sample_frac * self.environment.action_space.n)
        candidates = np.arange(1, self.environment.action_space.n)
        candidates = candidates[np.isin(candidates, targets.squeeze().detach().cpu().numpy(), invert=True)]

        sample = np.random.choice(candidates, (state.size(0), sample_size, self.metrics_k))
        sample = torch.as_tensor(sample, device=self.device)

        mask = torch.zeros_like(sample, dtype=torch.bool)
        mask[:, 0, 0] = 1

        actions = sample.masked_scatter(mask, targets)

        action_vectors = self.embedding(actions)
        action_vectors = action_vectors.flatten(-2)
        q_input = state_action_concat(state.detach(), action_vectors)

        q_expected = self.critic_local(q_input).squeeze(-1)

        t = torch.zeros_like(q_expected)
        t[:, 0] = 1

        critic_loss = F.mse_loss(q_expected, t, reduction="none").sum(1)
        loss = actor_loss + critic_loss
        loss = loss.mean()

        self.state_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.state_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return loss.item()
