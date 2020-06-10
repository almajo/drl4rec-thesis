import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from drlap.exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from models.RecommendationAgent import RecommendationAgent
from models.modules import FeedforwardNetwork
from util.KNN import state_action_concat
from util.decorators import timer
from util.sequential_replay import SequentialReplay, SequentialPriorityReplay


class DDPG(RecommendationAgent):
    """A DDPG Agent"""
    agent_name = "DDPG"

    def __init__(self, config):
        RecommendationAgent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        # Actor initialization
        self.actor_local, self.actor_target = self.get_actor(), self.get_actor()
        self.copy_model_over(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"],
                                          weight_decay=self.hyperparameters["Actor"]["weight_decay"])
        # Critic initialization
        self.critic_local, self.critic_target = self.get_critic(), self.get_critic()
        self.copy_model_over(self.critic_local, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"],
                                           weight_decay=self.hyperparameters["Critic"]["weight_decay"])

        if self.hyperparameters["buffer_type"] == "default":
            self.memory = SequentialReplay(config, config.seed, self.device)
        else:
            self.memory = SequentialPriorityReplay(config, config.seed, self.device)
        self.exploration_strategy = self.init_noise()

        self.output_scale_factor = self.embedding.weight.abs().max(dim=0).values.detach() \
            if self.hyperparameters["Embedding"]["type"] == "genome" else 1.

    def init_noise(self):
        return OU_Noise_Exploration(self.config)

    def get_action_size(self):
        return self.config.hyperparameters["Embedding"]["embedding_dim"]

    def post_pretrain_hook(self):
        self.embedding.weight.requires_grad_(False)
        self.copy_model_over(from_model=self.actor_local, to_model=self.actor_target)
        self.copy_model_over(from_model=self.critic_local, to_model=self.critic_target)

    def get_actor(self):
        return FeedforwardNetwork(input_dim=self.get_state_size(),
                                  hidden_dims=self.hyperparameters["Actor"]["linear_hidden_units"],
                                  output_dim=self.get_action_size(),
                                  output_fn=self.hyperparameters["Actor"]["final_layer_activation"]).to(self.device)

    def get_critic(self):
        return FeedforwardNetwork(input_dim=self.get_state_size() + self.get_action_size(),
                                  hidden_dims=self.hyperparameters["Critic"]["linear_hidden_units"],
                                  output_dim=1).to(self.device)

    def sample_experiences(self):
        return self.memory.sample()

    def _concat_with_sample_subset(self, state, action_id, sample_frac=0.2, return_candidates=False):
        """
        this version detaches the state, we are only upgrading the state through the actor
        """
        sample_size = int(sample_frac * self.environment.action_space.n)
        candidates = np.arange(1, self.environment.action_space.n)
        candidates = candidates[np.isin(candidates, action_id.squeeze().detach().cpu().numpy(), invert=True)]
        sample = np.random.default_rng(self.config.seed).choice(candidates, sample_size, replace=False, shuffle=False)
        sample = torch.as_tensor(sample, device=self.device)
        sample = sample.view(1, -1).expand(state.size(0), -1)
        actions = torch.cat([action_id, sample], dim=1)

        action_vectors = self.embedding(actions)
        cat_list = state_action_concat(state.detach(), action_vectors)
        if return_candidates:
            return cat_list, candidates
        return cat_list

    @timer
    def learn(self):
        importance_weights = None
        if isinstance(self.memory, SequentialPriorityReplay):
            states, actions, rewards, next_states, dones, importance_weights = self.sample_experiences()
        else:
            states, actions, rewards, next_states, dones = self.sample_experiences()
        with torch.no_grad():
            next_states, _ = self.create_state_vector(next_states)
        # Requires grad because it is also used in actor_loss
        current_state, _ = self.create_state_vector(states)
        self.critic_learn(current_state.detach(), actions, rewards, next_states, dones, importance_weights)

        targets = None
        if self.hyperparameters["batch_rl"]:
            targets = states.targets
        self.actor_learn(current_state, targets)

    def critic_learn(self, states, actions, rewards, next_states, dones, importance_weights):
        """Runs a learning iteration for the critic"""
        loss = self.compute_critic_loss(states, next_states, rewards, actions, dones, importance_weights)
        self.log_scalar("loss/critic", loss, global_step=self.episode_number)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_critic_loss(self, states, next_states, rewards, actions, dones, importance_weights):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = F.mse_loss(critic_expected, critic_targets, reduction='none').squeeze()
        if importance_weights is not None:
            self.memory.update_td_errors(loss.detach().cpu().numpy())
            loss = loss * importance_weights
        return loss.mean()

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    @timer
    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        actions_next = self.actor_target(next_states)
        critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        self.log_scalar("debug/expected_q_values", critic_expected.mean())
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
            "update_every_n_steps"] == 0

    @timer
    def actor_learn(self, states, targets):
        """Runs a learning iteration for the actor"""
        actor_loss = self.calculate_actor_loss(states, targets)
        self.log_scalar("loss/actor", actor_loss, global_step=self.episode_number)
        self.take_optimisation_step([self.state_optimizer, self.actor_optimizer], self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def calculate_actor_loss(self, states, targets):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        mse_loss = 0
        if targets is not None:
            t = self.embedding(targets).detach()
            mse_loss = F.mse_loss(actions_pred, t)
        proto_actions = self.output_scale_factor * actions_pred
        actor_loss = -self.critic_local(torch.cat((states, proto_actions), 1)).mean()

        return actor_loss + mse_loss

    def get_eval_action(self, obs, k):
        return self.pick_action(obs, eval_ep=True, top_k=k)

    def _load_model(self, parameter_dict):
        self.critic_local.load_state_dict(parameter_dict["critic_local"])
        self.critic_target.load_state_dict(parameter_dict["critic_target"])
        self.critic_optimizer.load_state_dict(parameter_dict["critic_optimizer"])
        self.actor_local.load_state_dict(parameter_dict["actor_local"])
        self.actor_target.load_state_dict(parameter_dict["actor_target"])
        self.actor_optimizer.load_state_dict(parameter_dict["actor_optimizer"])
