import random

import numpy as np

from drlap.exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy


class Epsilon_Greedy_Exploration(Base_Exploration_Strategy):
    """Implements an epsilon greedy exploration strategy"""

    def __init__(self, config):
        super().__init__(config)
        self.notified_that_exploration_turned_off = False
        if "exploration_cycle_episodes_length" in self.config.hyperparameters.keys():
            print("Using a cyclical exploration strategy")
            self.exploration_cycle_episodes_length = self.config.hyperparameters["exploration_cycle_episodes_length"]
        else:
            self.exploration_cycle_episodes_length = None
        # self.eps_config = config.hyperparameters.get("Epsilon")
        # self.m = (self.eps_config.get("end_value") - self.eps_config.get("start_value")) / self.eps_config.get(
        #     "end_episode")
        self.env = config.environment
        self.constant_epsilon = config.hyperparameters.get("static_epsilon")

        self.epsilon_set_size = min(self.config.hyperparameters.get("epsilon_sample_size"),
                                    self.config.environment.action_space.n)

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_fn = action_info["action_fn"]
        turn_off_exploration = action_info["turn_off_exploration"]
        sample_shape = action_info.get("sample_shape")
        if self.constant_epsilon is not None:
            epsilon = self.constant_epsilon
        else:
            epsilon = self.update_epsilon(action_info.get("episode_number"))
        if random.random() > epsilon or turn_off_exploration:
            return action_fn()

        return self.random_sample(sample_shape)

    def random_sample(self, k):
        sample_subset = self.env.get_sample_subset(self.epsilon_set_size, return_tensor=False)
        perm = np.random.permutation(self.epsilon_set_size)[:k]
        return sample_subset[..., perm]

    def update_epsilon(self, episode_number):
        if episode_number > self.eps_config.get("end_episode"):
            return self.eps_config.get("end_value")
        return self.m * episode_number + self.eps_config.get("start_value")

    def reset(self):
        """Resets the noise process"""
        pass
