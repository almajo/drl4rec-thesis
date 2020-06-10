import logging
import time
import warnings
from collections import deque

import numpy as np
import torch
from nn_builder.pytorch.NN import NN
from torch.nn import Module

warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter


class BaseAgent(object):

    def __init__(self, config):
        self.logger = self.setup_logger()
        self.debug_mode = config.debug_mode
        self.tb_writer = SummaryWriter(config.tb_dir) if config.tb_dir and config.use_tb else None
        self.tb_log_interval = config.tb_log_interval
        self.config = config
        self.environment = config.environment
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.config.action_size = self.action_size
        self.hyperparameters = config.hyperparameters
        self.total_episode_score_so_far = 0
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU and torch.cuda.is_available() else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        self.last_episode_returns = deque(maxlen=100)
        self.rolling_score_window = 20

    @property
    def is_train(self):
        return all(map(lambda y: y.training, filter(lambda x: isinstance(x, Module), self.__dict__.values())))

    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def log_scalar(self, name, value, global_step=None, interval=None):
        if global_step is None:
            global_step = self.episode_number
        interval = interval if interval else self.tb_log_interval
        if global_step % interval == 0 and self.tb_writer and global_step > 1:
            self.tb_writer.add_scalar(name, value, global_step=global_step)

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "overwrite_action_size" in self.config.__dict__:
            return self.config.overwrite_action_size
        if "action_size" in self.environment.__dict__:
            return self.environment.action_size
        if self.action_types == "DISCRETE":
            return self.environment.action_space.n
        else:
            return self.environment.action_space.shape[0]

    def eval(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.eval()

    def to_train(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.train()

    def setup_logger(self):
        return logging.getLogger()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.last_done = np.zeros(self.environment.num_envs, dtype=np.bool)
        if "exploration_strategy" in self.__dict__.keys():
            self.exploration_strategy.reset()

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes_to_run
        self.durations = deque([], maxlen=20)
        start = time.time()
        while self.episode_number < num_episodes:
            ep_start = time.time()
            self.reset_game()

            if self.state is None:
                break
            self.step()
            if save_and_print_results:
                self.save_and_print_result()
            self.durations.append(time.time() - ep_start)
            if self.config.save_model and self.episode_number % self.config.model_save_interval == 0:
                self.locally_save_policy()

        time_taken = time.time() - start
        if self.config.save_model:
            self.locally_save_policy()
        return self.last_episode_returns, self.last_episode_returns, time_taken

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, self.info = self.environment.step(action)
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def save_and_print_result(self, scope="train"):
        """Saves and prints results of the game"""
        self.save_result(scope)
        return self.print_rolling_result()

    def save_result(self, scope="train"):
        """Saves the result of an episode of the game"""
        self.last_episode_returns.append(self.total_episode_score_so_far)
        if self.durations and list(self.last_episode_returns):
            if self.episode_number % self.tb_log_interval == 0 and self.tb_writer and self.episode_number > 1:
                self.tb_writer.add_scalar("{}/running_return".format(scope),
                                          np.mean(list(self.last_episode_returns)[-self.rolling_score_window:]),
                                          global_step=self.episode_number)
                self.tb_writer.add_scalar("{}/time/sliding_episode_average".format(scope), np.mean(self.durations),
                                          global_step=self.episode_number)
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.total_episode_score_so_far > self.max_episode_score_seen:
            self.max_episode_score_seen = self.total_episode_score_so_far

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        return {"Return": self.total_episode_score_so_far,
                "Max Return Seen": self.max_episode_score_seen,
                "Rolling Return": np.mean(list(self.last_episode_returns)[-self.rolling_score_window:])}

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return self.memory.ready_to_learn(self.hyperparameters["batch_size"]) and \
               self.episode_number >= self.hyperparameters["min_steps_before_learning"]

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    @staticmethod
    def take_optimisation_step(optimizers, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optim in optimizers:
            optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), clipping_norm)
        for optim in optimizers:
            optim.step()

    def log_gradient_and_weight_information(self, network, optimizer):
        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None,
                  hyperparameters=None) -> torch.nn.Module:
        """Creates a neural network for the models to use"""
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        if key_to_use:
            hyperparameters = hyperparameters[key_to_use]
        if override_seed:
            seed = override_seed
        else:
            seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed)

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    @staticmethod
    def freeze_all_but_output_layers(network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(
                param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    @staticmethod
    def unfreeze_all_layers(network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
