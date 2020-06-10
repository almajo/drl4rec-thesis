import torch

from drlap.exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
from drlap.utilities.OU_Noise import OU_Noise


class OU_Noise_Exploration(Base_Exploration_Strategy):
    """Ornstein-Uhlenbeck noise process exploration strategy"""

    def __init__(self, config):
        super().__init__(config)
        self.noise = OU_Noise(self.config.action_size, self.config.seed, self.config.hyperparameters["mu"],
                              self.config.hyperparameters["theta"], self.config.hyperparameters["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        noise = self.noise.sample()
        action += torch.as_tensor(noise, device=action.device, dtype=torch.float32)
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
