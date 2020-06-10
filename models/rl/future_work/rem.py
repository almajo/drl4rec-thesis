import numpy as np
import torch
import torch.nn.functional as F

from models.rl.q_learning import ParameterizedDDQNAgent


class REM(ParameterizedDDQNAgent):
    agent_name = "REM"

    def __init__(self, config):
        super().__init__(config)
        self.q_loss_fn = F.smooth_l1_loss

    def get_q_network_and_optimizer(self):
        qn = self.create_network(state_size=self.state_size + self.embedding_dim,
                                 hidden_units=self.hyperparameters["linear_hidden_units"],
                                 output_dim=self.hyperparameters["num_heads"])
        adam = torch.optim.Adam(qn.parameters(),
                                lr=self.hyperparameters["learning_rate"],
                                weight_decay=self.hyperparameters["weight_decay"])
        return qn, adam

    def _get_q_weighted_average(self, q_values):
        uniforms = torch.rand_like(q_values)
        probs = uniforms / uniforms.sum(dim=-1, keepdim=True)

        new_q = q_values * probs
        new_q_estimate = new_q.sum(dim=-1)
        return new_q_estimate

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network.
        Local network computes the inner-argmax and target-network estimates the q_value for that
        """
        state_action = self.state_action_concat(next_states)
        max_action_indexes = self.q_network_local(state_action)
        weighted_q_values = self._get_q_weighted_average(max_action_indexes)
        vals = self.mask_seen_items(weighted_q_values)
        argmax_per_batch_index = vals.argmax(dim=1).squeeze(-1)

        state_action_new = state_action[torch.arange(state_action.size(0)), argmax_per_batch_index]
        q_targets_next = self.q_network_target(state_action_new)
        weighted_q_targets = self._get_q_weighted_average(q_targets_next)
        return weighted_q_targets

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        emb = self.embedding(actions.squeeze(-1))
        state_action = torch.cat([states, emb], dim=-1)
        Q_expected = self.q_network_local(state_action)
        weighted_q_values = self._get_q_weighted_average(Q_expected)
        return weighted_q_values

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        state = self.state_action_concat(state)
        a = self.q_network_local(state)
        a = a.mean(dim=-1)
        a = self.mask_seen_items(a)
        a[..., 0] = - float("inf")
        a = a.topk(k=top_k, dim=-1).indices
        return a

    def mask_seen_items(self, q_values: torch.Tensor):
        if self.masking_enabled:
            mask = self.user_history_mask_items
            q_values.scatter_(1, mask.to(q_values.device), float('-inf'))
        return q_values
