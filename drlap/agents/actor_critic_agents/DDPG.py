from copy import copy

import torch
import torch.nn.functional as functional
from torch import optim

from drlap.exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
from drlap.utilities.data_structures.Replay_Buffer import Replay_Buffer
from models.BaseAgent import BaseAgent
from util.decorators import timer


class DDPG(BaseAgent):
    """A DDPG Agent"""
    agent_name = "DDPG"

    def __init__(self, config, actor_output_dim=None, action_emb=None):
        BaseAgent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        action_out = actor_output_dim or self.action_size
        critic_input = action_emb or self.action_size

        self.critic_local = self.create_NN(input_dim=self.state_size + critic_input, output_dim=1,
                                           key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + critic_input, output_dim=1,
                                            key_to_use="Critic")
        BaseAgent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        replay_buffer = Replay_Buffer
        if "replay_buffer" in self.hyperparameters["Critic"]:
            replay_buffer = self.hyperparameters["Critic"]["replay_buffer"]
        self.memory = replay_buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=action_out, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=action_out, key_to_use="Actor")
        BaseAgent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = self.init_noise()

    def init_noise(self):
        return OU_Noise_Exploration(self.config)

    def locally_save_policy(self):
        try:
            env_id = self.environment.spec.id
        except AttributeError:
            env_id = self.environment.unwrapped.spec.id

        agents = {k: v.state_dict() for k, v in self.__dict__.items() if isinstance(v, torch.nn.Module)}
        config = copy(self.config)
        config.environment = env_id
        config.test_environment = env_id
        d = {
            "agent_name": self.agent_name,
            "config": config
        }
        d.update(agents)
        print("Saved model to {}".format(config.file_to_save_model))
        torch.save(d, config.file_to_save_model)

    def restore_parameters(self, data):
        self.critic_local.load_state_dict(data["critic_local"])
        self.critic_target.load_state_dict(data["critic_target"])
        self.actor_local.load_state_dict(data["actor_local"])
        self.actor_target.load_state_dict(data["actor_target"])

    @timer
    def step(self):
        """Runs a step in the game"""
        while not self.done:
            # print("State ", self.state.shape)
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.save_experience()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def sample_experiences(self):
        return self.memory.sample()

    def wrap_action_output(self, action, state):
        return action

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None:
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()
        if self.is_train:
            action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        return self.wrap_action_output(action.squeeze(0), state)

    @timer
    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.tb_writer.add_scalar("train/critic/mse_loss", loss.mean().item(), global_step=self.episode_number)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"], scope="critic")
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)

        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
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
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
            "update_every_n_steps"] == 0

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        if self.done:  # we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"], scope="actor")
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss
