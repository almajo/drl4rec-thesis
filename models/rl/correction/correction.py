from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim

from models.RecommendationAgent import RecommendationAgent
from models.modules import CorrectionReinforceMainPolicy
from util.helpers import unwrap_state

"""
Implementation Details:
https://dl.acm.org/doi/abs/10.1145/3289600.3290999
Top-K Off-Policy Correction

Has an unfair advantage against all other baselines: directly uses the log data to learn the 
beta-head for importance weight. It builds a model of P(a | s) from log data via supervised learning.

Does not use any softmax approximation because we do not have that many items as they have in youtube.
Same for inference, there is no knn lookup.

variance reduction technique: Min clipping to e**3 (like in the paper)
Exploration: stochastic policy that samples from the full distribution

Hyperparameters:
Not one single parameter except the list size and the min_clipping value is stated ...
architecture however is fixed with state-rnn, one relu linear layer, followed by the softmax

"""
# Same as in the paper
CLIPPING_VALUE = np.e ** 3
LOG_MIN = 1e-8
PROB_MIN = 1e-16


class Beta(nn.Module):
    def __init__(self, input_size, output_size):
        super(Beta, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.optim = torch.optim.Adam(self.linear.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state, action):
        logits = self.linear(state)
        loss = self.criterion(logits, action)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return logits


class CorrectionReinforce(RecommendationAgent):
    def learn(self):
        pass

    agent_name = "Correction"

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = self.hyperparameters["linear_hidden_units"][0]
        self.policy = CorrectionReinforceMainPolicy(self.get_state_size(), hidden_dim, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=self.hyperparameters["learning_rate"],
                                    weight_decay=self.hyperparameters["weight_decay"])
        self.beta = Beta(hidden_dim, self.action_size).to(self.device)
        self.episode_rewards = [list() for _ in self.environment.envs]
        self.episode_log_probabilities = [list() for _ in self.environment.envs]
        self.ppo_weights = [list() for _ in self.environment.envs]
        self.corrections = [list() for _ in self.environment.envs]
        self.last_dones = np.zeros(len(self.environment.envs), dtype=np.bool)
        self.use_ppo = self.hyperparameters.get("ppo")
        if self.use_ppo:
            self.last_policy = deepcopy(self.policy).to(self.device)

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, self.info = self.environment.step(action)
        if isinstance(action, np.ndarray) and "item" in self.info:
            indices = np.array(self.info.get("item"))
            ind_mask = np.zeros(shape=(indices.shape[0],), dtype=np.long)
            mask = indices != -1
            ind_mask[mask] = indices[mask]
            self.reward_index_mask = ind_mask
        self.total_episode_score_so_far += self.reward.mean()
        if self.done:
            self.last_episode_returns.append(self.total_episode_score_so_far)
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_rewards = [list() for _ in self.environment.envs]
        self.episode_log_probabilities = [list() for _ in self.environment.envs]
        self.corrections = [list() for _ in self.environment.envs]
        self.episode_step_number = 0
        self.last_dones = np.zeros(len(self.environment.envs), dtype=np.bool)
        self.ppo_weights = [list() for _ in self.environment.envs]

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, log_probabilities, corrections, ppo_weight = self.pick_action_and_get_log_probabilities()
        self.conduct_action(action)
        self.store_log_probs_and_correction(log_probabilities, corrections, ppo_weight)
        self.store_action(action)
        self.last_dones = self.info.get("dones").astype(np.bool)

    def pick_action_and_get_log_probabilities(self, state=None):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        if state is None:
            state = self.state
        state = unwrap_state(state, device=self.device)
        state, current_targets = self.create_state_vector(state)

        action_logits, hidden_state = self.policy(state)
        beta_logits = self.beta(hidden_state.detach(), current_targets)

        with torch.no_grad():
            beta_probs = beta_logits.softmax(dim=-1)
            # need the prob_min because can not be 0 and large logits lead to tiny softmax
            beta_samples = torch.multinomial(beta_probs + PROB_MIN, self.k)
            beta_prob = beta_probs.gather(1, beta_samples)

        ppo_weight = None
        action_log_prob = action_logits.log_softmax(dim=-1)
        if self.use_ppo:
            with torch.no_grad():
                curr_samples = torch.multinomial(action_log_prob.exp() + PROB_MIN, self.k)
                curr_prob = action_log_prob.gather(1, curr_samples)

                action_logits_last, _ = self.last_policy(state)
                action_prob_last = action_logits_last.softmax(dim=-1).gather(1, curr_samples)

                ppo_weight = curr_prob.exp() / (action_prob_last + 1e-8)

        action_prob = action_log_prob.gather(1, beta_samples)
        correction = torch.clamp_max_(torch.exp(action_prob) / beta_prob, CLIPPING_VALUE).detach()
        return beta_samples.cpu().detach().numpy(), action_prob, correction, ppo_weight

    def store_log_probs_and_correction(self, log_probabilities, corrections, ppo_weight):
        """Stores the log probabilities of picked actions to be used for learning later"""
        env_indices = self.info.get("indices")
        log_probabilities = log_probabilities[~self.last_dones]
        corrections = corrections[~self.last_dones]
        action_mask = self.reward_index_mask
        log_probabilities = log_probabilities[torch.arange(log_probabilities.shape[0]), action_mask]
        corrections = corrections[torch.arange(corrections.shape[0]), action_mask]
        if self.use_ppo:
            ppo = ppo_weight[~self.last_dones]
            ppo = ppo[torch.arange(ppo.shape[0]), action_mask]
            for not_done_env_index, log_prob, corr, ppo_w in zip(env_indices, log_probabilities, corrections, ppo):
                self.episode_log_probabilities[not_done_env_index].append(log_prob)
                self.corrections[not_done_env_index].append(corr)
                self.ppo_weights[not_done_env_index].append(ppo_w)
        else:
            for not_done_env_index, log_prob, corr in zip(env_indices, log_probabilities, corrections):
                self.episode_log_probabilities[not_done_env_index].append(log_prob)
                self.corrections[not_done_env_index].append(corr)

    def store_action(self, action):
        """Stores the action picked"""
        self.action = action

    def store_reward(self):
        """Stores the reward picked"""
        env_indices = self.info.get("indices")
        for i, not_done_env in enumerate(env_indices):
            self.episode_rewards[not_done_env].append(self.reward[i])

    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        policy_loss = self.calculate_policy_loss_on_episode()
        if policy_loss is not None:
            self.log_scalar("loss/actor", policy_loss, global_step=self.episode_number)
            if self.use_ppo:
                self.equalise_policies()
            self.take_optimisation_step([self.optimizer, self.state_optimizer], self.policy, policy_loss,
                                        self.hyperparameters.get("gradient_clipping_norm"))

    def _ppo_clipping(self, value):
        return torch.clamp(value, min=1.0 - self.hyperparameters["clip_epsilon"],
                           max=1.0 + self.hyperparameters["clip_epsilon"])

    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.last_policy.parameters(), self.policy.parameters()):
            old_param.data.copy_(new_param.data)

    def calculate_policy_loss_on_episode(self):
        """Calculates the loss from an episode"""
        # """Calculates the loss from an episode. Could be more efficient but enough for a start"""
        gamma = self.hyperparameters["discount_rate"]
        rewards = [np.array(t) for t in self.episode_rewards]
        sum_of_batch_rewards = np.array([ret.sum() for ret in rewards])

        if not any(sum_of_batch_rewards):
            # do not compute the loss if all rewards were 0 anyway
            return None
        trajectories_with_pos_reward_index = np.flatnonzero(sum_of_batch_rewards)
        batch_returns = []
        for index in trajectories_with_pos_reward_index:
            traj = rewards[index]
            R = 0
            returns = []
            for r in traj[::-1]:
                R = r + gamma * R
                returns.insert(0, R)

            q = torch.as_tensor(returns, device=self.device)
            q = q / (q.max() + np.finfo(np.float32).eps.item())

            # q = (q - q.mean()) / (q.std() + np.finfo(np.float32).eps.item())
            batch_returns.append(q)

        log_probs = [torch.stack(t) for e, t in enumerate(self.episode_log_probabilities) if
                     e in trajectories_with_pos_reward_index]
        corrections = [torch.stack(t) for e, t in enumerate(self.corrections) if
                       e in trajectories_with_pos_reward_index]
        b_losses = []

        if self.use_ppo:
            ppo_weights = [torch.stack(t) for e, t in enumerate(self.ppo_weights) if
                           e in trajectories_with_pos_reward_index]
            for b_logs, b_ret, corr, ppo in zip(log_probs, batch_returns, corrections, ppo_weights):
                probs = torch.exp(b_logs)
                top_k_correction = self.k * (1 - probs) ** (self.k - 1)
                base_term = top_k_correction * corr * b_ret * b_logs
                trajectory_loss_1 = (base_term * ppo).sum()
                trajectory_loss_2 = (base_term * self._ppo_clipping(ppo)).sum()
                loss = - torch.min(trajectory_loss_1, trajectory_loss_2)
                b_losses.append(loss)
        else:
            for b_logs, b_ret, corr in zip(log_probs, batch_returns, corrections):
                probs = torch.exp(b_logs)
                top_k_correction = self.k * (1 - probs) ** (self.k - 1)
                trajectory_loss = - (top_k_correction * corr * b_ret * b_logs).sum()

                # trajectory_loss = - (corr * b_ret * b_logs).sum()
                b_losses.append(trajectory_loss)

        loss = torch.stack(b_losses).mean()
        return loss

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        pass

    def _load_model(self, parameter_dict):
        self.policy.load_state_dict(parameter_dict["policy"])
        self.optimizer.load_state_dict(parameter_dict["optimizer"])
        self.beta.load_state_dict(parameter_dict["beta"])

    def get_eval_action(self, obs, k):
        state, _ = self.create_state_vector(obs)
        action_probs = self.policy.predict(state)
        if self.masking_enabled:
            action_probs.scatter_(1, self.user_history_mask_items.to(action_probs.device), float("-inf"))
        action_probs[..., 0] = float("-inf")
        sampled_elements = torch.topk(action_probs, k).indices
        return sampled_elements.cpu().numpy()

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        pass
