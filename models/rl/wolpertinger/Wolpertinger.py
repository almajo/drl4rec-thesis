import numpy as np
import torch
import torch.nn.functional as F

from models.DDPG import DDPG
from util.KNN import matmul_lookup
from util.exploration import Epsilon_Greedy_Exploration
from util.helpers import isin_2d

"""
Wolpertinger Agent, Inheritance from DDPG (could also be TD3).
State Aggregator is learned jointly with the Critic-Update (in DDPG-class)
Embeddings are controlled to have a maxnorm of 1 which means they are forced to be in the hyper-unit circle

We use exact knn lookup via matmul (at least for smaller datasets)

"""


class Wolpertinger(DDPG):
    agent_name = "Wolpertinger"

    def __init__(self, config):
        super().__init__(config)
        self.action_size = self.embedding.embedding_dim
        self.exploration_strategy = self.init_noise()
        self.num_neighbors = int(self.hyperparameters["wolpertinger_k_frac"] * self.environment.action_space.n)
        self.last_q_log_episode_number = -1

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        a = self.actor_local(state)
        a = a * self.output_scale_factor
        a, _ = self.proto_to_real_action(a, state, top_k)
        return a

    def proto_to_real_action(self, proto, state, k):
        weights = self.embedding(torch.arange(self.environment.action_space.n, device=self.device)).detach()
        indices = matmul_lookup(proto, weights, self.num_neighbors)
        vectors = self.embedding(indices)
        max_q_indices, max_q = self.get_nn_with_highest_q(state, indices, vectors, k)
        return max_q_indices, max_q

    def get_nn_with_highest_q(self, state, indices, vectors, top_k):
        state = state.unsqueeze(1).expand(-1, vectors.size(1), -1)
        state_actions = torch.cat([state, vectors], dim=-1)
        q_values = self.critic_target(state_actions)
        if self.masking_enabled:
            q_values = self.mask_items(indices, q_values)
        max_q, max_q_indices = torch.topk(q_values, k=top_k, dim=1)
        max_q_indices = max_q_indices.squeeze(-1)
        real_indices = indices.gather(1, max_q_indices)
        return real_indices, max_q

    def mask_items(self, indices, q_values):
        mask = self.user_history_mask_items.to(indices.device)
        contained = isin_2d(indices, mask)
        q_values[contained.unsqueeze(-1)] = float("-inf")
        return q_values

    def compute_expected_critic_values(self, states, actions):
        actions = self.embedding(actions).detach()
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions.squeeze()), 1))

        if self.last_q_log_episode_number != self.episode_number:
            self.log_scalar("debug/expected_q_values", critic_expected.mean())
            self.last_q_log_episode_number = self.episode_number
        return critic_expected

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        proto_actions_next = self.actor_target(next_states)
        _, critic_targets_next = self.proto_to_real_action(proto_actions_next, next_states, k=1)

        return critic_targets_next.squeeze(-1)

    def init_noise(self):
        return Epsilon_Greedy_Exploration(self.config)

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute the ranking for the true item to be set on the first position and maximize it
        :param state:
        :param targets:
        :return:
        """
        action = self.actor_local(state)
        action = self.output_scale_factor * action
        target_actions = self.embedding(targets).detach()
        actor_loss = F.mse_loss(action, target_actions, reduction="none").sum(dim=1)

        sample_actions = self._concat_with_sample_subset(state, targets.unsqueeze(-1))
        q_expected = self.critic_local(sample_actions).squeeze(-1)

        t = torch.zeros_like(q_expected)
        t[:, 0] = 1

        critic_loss = F.mse_loss(q_expected, t, reduction="none").sum(1)

        loss = (actor_loss + critic_loss).mean()
        self.state_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()
        self.state_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return loss.item()
