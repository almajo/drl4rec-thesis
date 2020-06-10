import numpy as np
import torch
import torch.nn.functional as F

from models.rl.q_learning.q import DDQNAgent


class QRDQN(DDQNAgent):
    agent_name = "QR-DQN"

    def __init__(self, config):
        super().__init__(config)
        self.q_loss_fn = F.smooth_l1_loss
        self.num_quantiles = self.hyperparameters["num_quantiles"]
        self.kappa = 1

    def get_q_network_and_optimizer(self):
        qn = self.create_network(state_size=self.state_size,
                                 hidden_units=self.hyperparameters["linear_hidden_units"],
                                 output_dim=self.environment.action_space.n * self.hyperparameters["num_quantiles"],
                                 dropout=0)
        adam = torch.optim.Adam(qn.parameters(),
                                lr=self.hyperparameters["learning_rate"],
                                weight_decay=self.hyperparameters["weight_decay"])
        return qn, adam

    def network_forward(self, network, state):
        return network(state).view(-1, self.environment.action_space.n, self.hyperparameters["num_quantiles"])

    def compute_loss(self, states, next_states, rewards, actions, dones, importance_weights=None):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            target_theta = self.compute_q_targets(next_states, rewards, dones)
        theta = self.compute_expected_q_values(states, actions)

        if self.last_q_log_episode_number != self.episode_number:
            self.log_scalar("debug/mean_of_dist", theta.detach().mean())
            self.last_q_log_episode_number = self.episode_number

        target_distribution = target_theta.unsqueeze(1)

        bellman_errors = target_distribution - theta.unsqueeze(-1)

        huber_loss = (  # Eq. 9 of paper.
                (torch.abs(bellman_errors) <= self.kappa).float() *
                0.5 * bellman_errors ** 2 +
                (torch.abs(bellman_errors) > self.kappa).float() *
                self.kappa * (torch.abs(bellman_errors) - 0.5 * self.kappa))

        # huber_loss = F.smooth_l1_loss(theta, target_theta, reduction="none")
        tau_hat = (torch.arange(self.num_quantiles, dtype=torch.float32, device=self.device) + 0.5) / self.num_quantiles
        tau_hat = tau_hat[None, :, None]
        # no idea whats happening here
        q_huber_loss = huber_loss * torch.abs(tau_hat - (bellman_errors.detach() < 0).float())
        loss = q_huber_loss.mean(2).sum(1)

        if importance_weights is not None:
            self.memory.update_td_errors(loss.sqrt().detach().cpu().numpy())
            loss = loss * importance_weights
        return loss.mean()

    def compute_q_values_for_current_states(self, rewards, q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        q_targets_current = rewards + (self.hyperparameters["discount_rate"] * q_targets_next * (1 - dones))
        return q_targets_current

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network. Double DQN
        uses the local index to pick the maximum q_value action and then the target network to calculate the q_value.
        The reasoning behind this is that it will help stop the network from overestimating q values"""

        quantiles = self.network_forward(self.q_network_local, next_states)
        max_mean_values = quantiles.mean(2).argmax(1)

        target_quantiles = self.network_forward(self.q_network_target, next_states)
        action_target_values = target_quantiles[torch.arange(target_quantiles.size(0), device=target_quantiles.device),
                                                max_mean_values]

        return action_target_values

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        quantiles = self.network_forward(self.q_network_local, states)
        theta = quantiles[torch.arange(quantiles.size(0), device=quantiles.device), actions.squeeze()]
        return theta

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        a = self.network_forward(self.q_network_local, state)
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
