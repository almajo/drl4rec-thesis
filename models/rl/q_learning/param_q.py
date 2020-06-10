import numpy as np
import torch
import torch.nn.functional as F

from models.RecommendationAgent import RecommendationAgent
from models.modules import FeedforwardNetwork, ParamDuelling
from util.KNN import state_action_concat
from util.exploration import Epsilon_Greedy_Exploration
from util.sequential_replay import SequentialReplay, SequentialPriorityReplay


class ParameterizedDQNAgent(RecommendationAgent):
    agent_name = "PDQN"

    def __init__(self, config):
        super().__init__(config)
        self.q_network_local, self.q_network_optimizer = self.get_q_network_and_optimizer()
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)
        if self.hyperparameters["buffer_type"] == "default":
            self.memory = SequentialReplay(config, config.seed, self.device)
        else:
            self.memory = SequentialPriorityReplay(config, config.seed, self.device)
        self.last_q_log_episode_number = -1

        self.q_loss_fn = F.mse_loss

    def get_q_network_and_optimizer(self):
        qn = FeedforwardNetwork(input_dim=self.state_size + self.embedding_dim,
                                hidden_dims=self.hyperparameters["linear_hidden_units"],
                                output_dim=1).to(self.device)
        adam = torch.optim.Adam(qn.parameters(),
                                lr=self.hyperparameters["learning_rate"],
                                weight_decay=self.hyperparameters["weight_decay"])
        return qn, adam

    def mask_seen_items(self, q_values: torch.Tensor):
        if self.masking_enabled:
            mask = self.user_history_mask_items
            q_values.squeeze(-1).scatter_(1, mask.to(q_values.device), float('-inf'))
        return q_values

    def state_action_concat(self, state):
        if state.ndimension() == 2:
            state = state.unsqueeze(1)
        actions = self.embedding(
            torch.arange(self.embedding.num_embeddings, dtype=torch.long, device=self.device)).unsqueeze(0)
        actions = actions.expand(state.size(0), -1, -1)
        state = state.expand(-1, actions.size(1), -1)
        concat_vec = torch.cat([state, actions], dim=-1)
        return concat_vec

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        state = self.state_action_concat(state)
        a = self.q_network_local(state)
        a = self.mask_seen_items(a).squeeze(-1)
        a[..., 0] = - float("inf")
        a = a.topk(k=top_k, dim=-1).indices
        return a

    def learn(self):
        """Runs a learning iteration for the Q network"""
        importance_weights = None
        if isinstance(self.memory, SequentialPriorityReplay):
            states, actions, rewards, next_states, dones, importance_weights = self.memory.sample()
        else:
            states, actions, rewards, next_states, dones = self.memory.sample()
        current_state, current_targets = self.create_state_vector(states)
        # We compute the next_state_vector last because it sets the state history variable which is used for masking
        with torch.no_grad():
            next_states, next_targets = self.create_state_vector(next_states)

        loss = self.compute_loss(current_state, next_states, rewards, actions, dones,
                                 importance_weights=importance_weights)
        if self.last_q_log_episode_number != self.episode_number:
            self.log_scalar("loss/critic", loss)
            self.last_q_log_episode_number = self.episode_number
        self.take_optimisation_step([self.q_network_optimizer, self.state_optimizer], self.q_network_local, loss,
                                    self.hyperparameters.get("gradient_clipping_norm"))

    def compute_loss(self, states, next_states, rewards, actions, dones, importance_weights=None):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            q_targets = self.compute_q_targets(next_states, rewards, dones)
        q_expected = self.compute_expected_q_values(states, actions)
        if self.last_q_log_episode_number != self.episode_number:
            self.log_scalar("debug/expected_q_values", q_expected.mean())

        q_targets = q_targets.squeeze()
        q_expected = q_expected.squeeze()

        loss = self.q_loss_fn(q_expected, q_targets, reduction="none")
        if importance_weights is not None:
            self.memory.update_td_errors(loss.detach().cpu().numpy())
            loss = loss * importance_weights
        return loss.mean()

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        q_targets_next = self.compute_q_values_for_next_states(next_states)
        q_targets = self.compute_q_values_for_current_states(rewards, q_targets_next, dones)
        return q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        state_action = self.state_action_concat(next_states)
        q_targets = self.q_network_local(state_action)
        q_targets = self.mask_seen_items(q_targets)
        max_q_targets = q_targets.max(1)[0]
        return max_q_targets.unsqueeze(-1)

    def compute_q_values_for_current_states(self, rewards, q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        if q_targets_next.size(-1) != 1:
            q_targets_next = q_targets_next.unsqueeze(-1)
        q_targets_current = rewards + (self.hyperparameters["discount_rate"] * q_targets_next * (1 - dones))
        return q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        emb = self.embedding(actions.squeeze(-1))
        state_action = torch.cat([states, emb], dim=-1)
        Q_expected = self.q_network_local(state_action)
        return Q_expected.squeeze(-1)

    def get_eval_action(self, obs, k):
        return self.pick_action(obs, eval_ep=True)

    def _load_model(self, parameter_dict):
        self.q_network_local.load_state_dict(parameter_dict["q_network_local"])
        self.q_network_optimizer.load_state_dict(parameter_dict["q_network_optimizer"])

    def _concat_with_sample_subset(self, state, action_id, sample_frac=0.2):
        sample_size = int(sample_frac * self.environment.action_space.n)

        candidates = np.arange(1, self.environment.action_space.n)
        candidates = candidates[np.isin(candidates, action_id.squeeze().detach().cpu().numpy(), invert=True)]
        sample = np.random.default_rng(self.config.seed).choice(candidates, sample_size, replace=False, shuffle=False)
        sample = torch.as_tensor(sample, device=self.device)
        sample = sample.view(1, -1).expand(state.size(0), -1)
        actions = torch.cat([action_id, sample], dim=1)

        action_vectors = self.embedding(actions)
        return state_action_concat(state, action_vectors)

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        sample_actions = self._concat_with_sample_subset(state, targets.unsqueeze(-1))
        q_expected = self.q_network_local(state, sample_actions).squeeze(-1)

        t = torch.zeros_like(q_expected)
        t[:, 0] = 1

        loss = F.mse_loss(q_expected, t, reduction="none").sum(1).mean()

        self.state_optimizer.zero_grad()
        self.q_network_optimizer.zero_grad()
        loss.backward()
        self.state_optimizer.step()
        self.q_network_optimizer.step()

        return loss.item()


class ParameterizedDDQNAgent(ParameterizedDQNAgent):
    agent_name = "PDDQN"

    def __init__(self, config):
        super().__init__(config)
        self.q_network_target, _ = self.get_q_network_and_optimizer()
        self.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def post_pretrain_hook(self):
        self.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)

    def learn(self):
        """Runs a learning iteration for the Q network"""
        super().learn()
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target,
                                           self.hyperparameters["tau"])  # Update the target network

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network.
        Local network computes the inner-argmax and target-network estimates the q_value for that
        """
        state_action = self.state_action_concat(next_states)
        max_action_indexes = self.q_network_local(state_action)
        vals = self.mask_seen_items(max_action_indexes)
        argmax_per_batch_index = vals.argmax(dim=1).squeeze(-1)

        state_action_new = state_action[np.arange(state_action.size(0)), argmax_per_batch_index]
        q_targets_next = self.q_network_target(state_action_new).squeeze(-1)
        return q_targets_next

    def _load_model(self, parameter_dict):
        super()._load_model(parameter_dict)
        self.q_network_target.load_state_dict(parameter_dict["q_network_target"])


class ParameterizedDuellingDDQNAgent(ParameterizedDDQNAgent):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    agent_name = "PDDDQN"

    def __init__(self, config):
        super().__init__(config)

    def get_q_network_and_optimizer(self):
        net = ParamDuelling(self.state_size, self.embedding_dim, self.hyperparameters["linear_hidden_units"]).to(
            self.device)
        adam = torch.optim.Adam(net.parameters(),
                                lr=self.hyperparameters["learning_rate"],
                                weight_decay=self.hyperparameters["weight_decay"])
        return net, adam

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network.
        Local network computes the inner-argmax and target-network estimates the q_value for that
        """
        state_action = self.state_action_concat(next_states)
        max_action_indexes = self.q_network_local(next_states, state_action)
        vals = self.mask_seen_items(max_action_indexes)
        argmax_per_batch_index = vals.argmax(dim=1, keepdim=True).squeeze(-1)

        sample_actions = self._concat_with_sample_subset(next_states, argmax_per_batch_index)
        q_targets_next = self.q_network_target(next_states, sample_actions)
        q_targets_next = q_targets_next[:, 0]
        # q_targets_next = q_targets_next.gather(1, argmax_per_batch_index)
        return q_targets_next

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        # state_actions = self.state_action_concat(states)
        sample_actions = self._concat_with_sample_subset(states, actions)
        q_expected = self.q_network_local(states, sample_actions)

        q_expected = q_expected[:, 0]
        # q_expected = q_expected.gather(1, actions)
        return q_expected

    def get_eval_action(self, obs, k):
        return self.pick_action(obs, eval_ep=True)

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        state_action = self.state_action_concat(state)
        a = self.q_network_local.predict(state_action).squeeze(-1)
        a = self.mask_seen_items(a)
        a[..., 0] = - float("inf")
        a = a.topk(k=top_k, dim=-1).indices
        return a
