import numpy as np
import torch
import torch.nn.functional as F

from models.modules import FeedforwardNetwork, DuelingPolicy
from models.rl.q_learning import ParameterizedDQNAgent


class DQNAgent(ParameterizedDQNAgent):
    agent_name = "DQN"

    def __init__(self, config):
        super().__init__(config)

    def get_q_network_and_optimizer(self):
        qn = FeedforwardNetwork(input_dim=self.state_size,
                                hidden_dims=self.hyperparameters["linear_hidden_units"],
                                output_dim=self.action_size).to(self.device)
        adam = torch.optim.Adam(qn.parameters(),
                                lr=self.hyperparameters["learning_rate"],
                                weight_decay=self.hyperparameters["weight_decay"])
        return qn, adam

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        at = self.q_network_local(state)
        at = self.mask_seen_items(at)
        at = at.topk(k=top_k).indices
        return at

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        q_targets_next = self.q_network_local(next_states)
        q_targets_next = self.mask_seen_items(q_targets_next)
        max_q_values = q_targets_next.max(dim=-1, keepdim=True)[0]
        return max_q_values

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        q_expected = self.q_network_local(states)
        expected = q_expected.gather(1, actions)

        return expected

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        a = self.q_network_local(state)

        t = torch.zeros_like(a)
        t[torch.arange(t.size(0)), targets] = 1

        loss = F.mse_loss(a, t, reduction="none").sum(1).mean()

        self.state_optimizer.zero_grad()
        self.q_network_optimizer.zero_grad()
        loss.backward()
        self.state_optimizer.step()
        self.q_network_optimizer.step()

        return loss.item()


class DDQNAgent(DQNAgent):
    agent_name = "DDQN"

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
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""
        max_action_indexes = self.q_network_local(next_states)
        max_action_indexes = self.mask_seen_items(max_action_indexes).detach().argmax(1)
        q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return q_targets_next

    def _load_model(self, parameter_dict):
        super()._load_model(parameter_dict)
        self.q_network_target.load_state_dict(parameter_dict["q_network_target"])


class Dueling(DDQNAgent):
    """A dueling double DQN agent as described in the paper http://proceedings.mlr.press/v48/wangf16.pdf"""
    agent_name = "Dueling"

    def __init__(self, config):
        super().__init__(config)

    def get_q_network_and_optimizer(self):
        qn = DuelingPolicy(self.state_size,
                           self.hyperparameters["linear_hidden_units"],
                           self.environment.action_space.n).to(self.device)
        adam = torch.optim.Adam(qn.parameters(),
                                lr=self.hyperparameters["learning_rate"],
                                weight_decay=self.hyperparameters["weight_decay"])
        return qn, adam

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        at = self.q_network_local.predict(state)
        at = self.mask_seen_items(at)
        at[..., 0] = - float("inf")
        at = at.topk(k=top_k).indices
        return at

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""

        # as we are only interested in the max, we can do the predict path only
        max_action_indexes = self.q_network_local.predict(next_states)
        max_action_indexes = self.mask_seen_items(max_action_indexes).argmax(1, keepdim=True)

        # here we need the full policy because we are interested in the actual values
        q_values = self.q_network_target(next_states)
        q_targets_next = q_values.gather(1, max_action_indexes)
        return q_targets_next

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        q_values = self.q_network_local(states)
        q_exp = q_values.gather(1, actions)
        return q_exp
