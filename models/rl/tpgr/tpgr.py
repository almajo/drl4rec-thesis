import logging

import numpy as np
import torch

from models.RecommendationAgent import RecommendationAgent
from models.modules import FeedforwardNetwork
from models.rl.correction import CLIPPING_VALUE
from models.rl.tpgr import TreeAgent
from models.rl.tpgr import create_clustering_tree
from util.decorators import timer
from util.helpers import unwrap_state, numpy_isin_2d

"""
Implementation details of TPGR

we are creating the pca tree from the original implementation as a PCA hierarchical clustering.
It is hard coded for the depth 2 case which we are only using
"""


class TPGR(RecommendationAgent):
    def learn(self):
        pass

    agent_name = "TPGR"

    def __init__(self, config):
        RecommendationAgent.__init__(self, config)
        logging.info("Creating clustering tree")
        self.tree = create_clustering_tree(self.environment,
                                           tree_depth=self.config.hyperparameters["tree_depth"])
        logging.info("Initializing actors")
        self.agent = TreeAgent(self.tree, self.get_state_size(),
                               self.hyperparameters["linear_hidden_units"],
                               output_list_size=self.metrics_k).to(self.device)
        self.off_policy_agent = FeedforwardNetwork(input_dim=self.get_state_size(),
                                                   hidden_dims=[256],
                                                   output_dim=self.environment.action_space.n).to(self.device)
        self.num_items = self.environment.action_space.n
        self.num_children = self.tree.child_num
        self.depth = self.tree.bc_dim
        self.sample_k = min(self.metrics_k, int(self.tree.child_num / 2))  # Puffer for empty samples

        self.optimizer = torch.optim.Adam(self.agent.parameters(),
                                          lr=self.hyperparameters["learning_rate"],
                                          weight_decay=self.hyperparameters["weight_decay"])
        self.off_p_optimizer = torch.optim.Adam(self.off_policy_agent.parameters(),
                                                lr=self.hyperparameters["learning_rate"],
                                                weight_decay=self.hyperparameters["weight_decay"])
        self.episode_actions = [list() for _ in self.environment.envs]
        self.last_dones = np.zeros(len(self.environment.envs), dtype=np.bool)
        self.episode_states = []
        self.corrections = [list() for _ in self.environment.envs]

    def get_eval_action(self, obs, k):
        state, _ = self.create_state_vector(obs)
        action_trajectory, _ = self.agent(state, deterministic=True, mask_items=self.user_history_mask_items)
        action = self.action_trajectory_to_action(action_trajectory)
        if self.masking_enabled:
            action, _ = self.mask_action_output(action)
        return action

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.episode_states.append(self.state)
        self.next_state, self.reward, self.done, self.info = self.environment.step(action)
        if isinstance(action, np.ndarray) and "item" in self.info:
            indices = np.array(self.info.get("item"))
            ind_mask = np.zeros_like(indices)
            mask = indices != -1
            ind_mask[mask] = indices[mask]
            self.reward_index_mask = ind_mask
        self.total_episode_score_so_far += self.reward.mean()
        if self.done:
            self.last_episode_returns.append(self.total_episode_score_so_far)
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def action_trajectory_to_action(self, action_trajectory):
        def fn(array):
            return self.tree.get_action_to_trajectory(tuple(array))

        action = np.apply_along_axis(fn, -1, action_trajectory)
        return action

    @timer
    def mask_action_output(self, actions):
        mask = self.user_history_mask_items
        if not isinstance(mask, np.ndarray):
            mask = self.user_history_mask_items.numpy()
        bool_mask = numpy_isin_2d(actions, mask)
        actions[bool_mask] = 0
        return actions, bool_mask

    def pick_action_and_log_probs(self, state=None):
        if state is None:
            state = self.state
        state = unwrap_state(state, device=self.device)
        state, targets = self.create_state_vector(state)
        if self.hyperparameters["batch_rl"]:
            # Get the log probs for the target policy
            actions, action_log_probs = self.agent.log_probs_for_actions(state, targets)

            # Update the off-policy network for IS weights (behavior policy approximation)
            beta_logits = self.off_policy_agent(state.detach())
            beta_log_probs = beta_logits.log_softmax(dim=-1)
            beta_log_probs = beta_log_probs[torch.arange(beta_log_probs.size(0)), targets]
            self.update_off_policy_agent(beta_log_probs)

            is_weights = torch.clamp_max_(torch.exp(action_log_probs) / torch.exp(beta_log_probs),
                                          CLIPPING_VALUE).detach()
            return actions, action_log_probs, is_weights

        action_trajectory, action_log_probs = self.agent(state, deterministic=False)
        actions = self.action_trajectory_to_action(action_trajectory)
        # if self.masking_enabled:
        #     actions, mask = self.mask_action_output(actions)
        return actions, action_log_probs, None

    def calculate_policy_loss_on_episode(self):
        """Calculates the loss from an episode"""
        # """Calculates the loss from an episode. Could be more efficient but enough for a start"""
        gamma = self.hyperparameters["discount_rate"]
        rewards = [np.array(t) for t in self.episode_rewards]
        sum_of_batch_rewards = np.array([ret.sum() for ret in rewards])
        if not any(sum_of_batch_rewards):
            return
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
        if len(self.corrections) and self.hyperparameters.get("batch_rl"):
            corrections = [torch.stack(t) for e, t in enumerate(self.corrections) if
                           e in trajectories_with_pos_reward_index]
        else:
            corrections = [1] * len(log_probs)
        b_losses = []
        for log_probs, q_values, is_weight in zip(log_probs, batch_returns, corrections):
            trajectory_loss = - (is_weight * log_probs * q_values).sum()
            b_losses.append(trajectory_loss)

        loss = torch.stack(b_losses).mean()
        return loss

    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        policy_loss = self.calculate_policy_loss_on_episode()
        if policy_loss is not None:
            self.log_scalar("loss/actor", policy_loss, global_step=self.episode_number)
            self.take_optimisation_step([self.optimizer, self.state_optimizer], self.agent, policy_loss,
                                        self.hyperparameters.get("gradient_clipping_norm"))

    def update_off_policy_agent(self, log_probs):
        off_policy_loss = - log_probs.mean()
        self.log_scalar("loss/off_policy_actor", off_policy_loss, global_step=self.episode_number)
        self.take_optimisation_step(self.off_p_optimizer, self.off_policy_agent, off_policy_loss,
                                    self.hyperparameters.get("gradient_clipping_norm"))

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        actions, log_probs, is_weights = self.pick_action_and_log_probs()
        self.conduct_action(actions)
        if is_weights is None:
            self.store_log_probs(log_probs)
        else:
            self.store_log_probs_and_correction(log_probs, is_weights)

        self.last_dones = self.info.get("dones").astype(np.bool)

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
        self.episode_actions = [list() for _ in self.environment.envs]
        self.episode_states = []
        self.last_dones = np.zeros(len(self.environment.envs), dtype=np.bool)

    def store_log_probs(self, log_probabilities):
        """Stores the log probabilities of picked actions to be used for learning later"""
        env_indices = self.info.get("indices")
        log_probabilities = log_probabilities[~self.last_dones]
        action_mask = self.reward_index_mask
        log_probabilities = log_probabilities[
            torch.arange(log_probabilities.shape[0], device=log_probabilities.device), action_mask]
        for not_done_env_index, log_prob in zip(env_indices, log_probabilities):
            self.episode_log_probabilities[not_done_env_index].append(log_prob)

    def store_log_probs_and_correction(self, log_probabilities, corrections):
        """Stores the log probabilities of picked actions to be used for learning later"""
        env_indices = self.info.get("indices")
        log_probabilities = log_probabilities[~self.last_dones]
        corrections = corrections[~self.last_dones]
        action_mask = self.reward_index_mask
        log_probabilities = log_probabilities[torch.arange(log_probabilities.shape[0]), action_mask]
        corrections = corrections[torch.arange(corrections.shape[0]), action_mask]
        for not_done_env_index, log_prob, corr in zip(env_indices, log_probabilities, corrections):
            self.episode_log_probabilities[not_done_env_index].append(log_prob)
            self.corrections[not_done_env_index].append(corr)

    def store_reward(self):
        """Stores the reward picked"""
        env_indices = self.info.get("indices")
        for i, not_done_env in enumerate(env_indices):
            self.episode_rewards[not_done_env].append(self.reward[i])

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done

    def _load_model(self, parameter_dict):
        old_version = parameter_dict.get("policies")
        if old_version is not None:
            # compatibility function for old architecture
            self.agent.policies.load_state_dict(old_version)
        else:
            self.agent.load_state_dict(parameter_dict.get("agent"))

        if "off_policy_agent" in parameter_dict:
            self.off_policy_agent.load_state_dict(parameter_dict.get("off_policy_agent"))
        self.optimizer.load_state_dict(parameter_dict["optimizer"])

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        """
        this will be called for every transition. However we only store the given transitions until the episode is done.
        Then we do the PG update
        :param state:
        :param targets:
        :param done:
        :return:
        """
        targets, log_probs = self.agent.log_probs_for_actions(state, targets)
        loss = - log_probs.mean()
        self.state_optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.state_optimizer.step()
        self.optimizer.step()

        return loss.item()

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        # TODO not used here
        pass
