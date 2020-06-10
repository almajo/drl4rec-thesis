import logging
from abc import ABC, abstractmethod
from collections import deque
from copy import copy, deepcopy
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from environments.recommendation_env import ReturnStateTuple
from evaluation.evaluation import Evaluation
from models.BaseAgent import BaseAgent
from models.StateAgg import RNNStateAgg
from models.modules import FeedforwardNetwork
from util.custom_embeddings import get_embedding
from util.decorators import timer
from util.exploration import Epsilon_Greedy_Exploration
from util.helpers import unwrap_state
from util.in_out import get_output_line
from util.model_saver import ModelSaver
from util.optimizers import MultiAdam
from util.sequential_replay import Buffer


# necessary workaround for storing embedding
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
#

class RecommendationAgent(BaseAgent, Evaluation, ABC):
    name = "Recommender"
    agent_name = "Recommender"

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        Evaluation.__init__(self, self.config.metrics_k)
        self.metrics_k = config.metrics_k
        self.state_config = self.config.hyperparameters["State"]
        self.embedding = self.get_embedding().to(self.device)
        self.embedding_dim = self.embedding.embedding_dim

        # Initialize state-module
        self.state_agg = RNNStateAgg(self.embedding,
                                     state_config=self.state_config,
                                     reward_range=[0, 1],
                                     with_rewards=False
                                     ).to(self.device)
        self.state_size = self.state_agg.state_size
        self.state_optimizer = self.create_state_optimizer()

        if self.config.hyperparameters["state-only-pretrain"] or self.config.hyperparameters["pretrain"]:
            save_dir = Path(config.file_to_save_model).parent / "pretrain"
            self.pretrain_model_saver = ModelSaver(save_dir)

        if self.config.hyperparameters["state-only-pretrain"]:
            self.output_layer = torch.nn.Linear(self.state_size, self.environment.action_space.n).to(self.device)
            self.output_layer_optimizer = torch.optim.Adam(self.output_layer.parameters())

        self.user_history_mask_items = None
        self.masking_enabled = self.hyperparameters.get("history_masking")

        self.model_saver = ModelSaver(Path(config.file_to_save_model).parent)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)

        self.last_done = np.zeros(self.environment.num_envs, dtype=np.bool)

        # if not self.model_saver.model_exists():
        #     self.tb_writer.add_embedding(self.embedding.weight)

    def get_embedding(self):
        return get_embedding(self.config.hyperparameters["Embedding"])

    def create_state_optimizer(self):
        if "Actor" in self.hyperparameters:
            m = MultiAdam(lr=self.hyperparameters["Actor"]["learning_rate"],
                          weight_decay=self.hyperparameters["Actor"].get("weight_decay"))
        else:
            m = MultiAdam(lr=self.hyperparameters["learning_rate"],
                          weight_decay=self.hyperparameters["weight_decay"])

        if self.state_agg.encoder is not None:
            m.add_dense_parameters(self.state_agg.encoder.parameters())

        if self.embedding.weight.requires_grad:
            params = self.embedding.parameters()
            if self.embedding.sparse:
                m.add_sparse_parameters(params)
            else:
                m.add_dense_parameters(params)
        return m

    @abstractmethod
    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        """
        Returns a tensor or numpy.ndarray after exploration
        """
        pass

    def pick_action(self, state=None, eval_ep=False, top_k=None):
        if top_k is None:
            top_k = self.k
        if state is None:
            state = self.state
        if self.global_step_number < self.hyperparameters["min_steps_before_learning"] and not eval_ep:
            return self.sample_from_action_space(num_items=self.metrics_k)
        with torch.no_grad():
            if not eval_ep:
                state = unwrap_state(state, device=self.device)
            state, targets = self.create_state_vector(state)
            self.eval()

            def action_fn():
                return self.state_to_action(state, eval_ep, top_k)

            action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_fn": action_fn,
                                                                                        "turn_off_exploration": eval_ep,
                                                                                        "episode_number": self.episode_number,
                                                                                        "sample_shape": top_k})
            self.to_train()
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()
        return action

    def pretrain_eval_fn(self, observation, k):
        state, _ = self.create_state_vector(observation)
        logits = self.output_layer(state)
        top_k_actions = logits.topk(k)
        return top_k_actions.indices.detach().cpu().numpy()

    @abstractmethod
    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        pass

    def pretrain_from_batch(self, observations):
        targets = observations.targets
        state, _ = self.create_state_vector(observations)

        loss = self._supervised_learning_from_batch(state, targets)
        return loss

    def pretrain_state_from_batch(self, observations) -> float:
        """
        Returns (error, top-k actions)-tuple from a batch of observations and performs a loss optimization in it
        @param observations: ReturnStateTuple
        """

        targets = observations.targets
        state, _ = self.create_state_vector(observations)
        logits = self.output_layer(state)
        error = cross_entropy(logits, targets)

        self.output_layer_optimizer.zero_grad()
        self.state_optimizer.zero_grad()
        error.backward()
        self.state_optimizer.step()
        self.output_layer_optimizer.step()

        return error.item()

    def save_state_module(self):
        logging.info(get_output_line())
        save_path = self.config.hyperparameters["State"].get("save_path")
        if save_path is not None and not save_path.exists():
            logging.info("Saving RNN and start-embeddings from disk")
            d = {
                "rnn": self.state_agg.encoder.state_dict(),
                "optimizer": self.state_optimizer.state_dict(),
                "embedding": self.embedding.state_dict()
            }

    def pretrain(self):
        """
        Pretraining the actor in a supervised-fashion
        """
        batch_size = self.hyperparameters["batch_size"]
        buffer = Buffer(batch_size, int(1e6))

        logging.info("Filling the train_set")
        env = deepcopy(self.environment)
        for e in env.envs:
            e.num_repeats = 1
        while 1:
            obs = env.reset()
            if obs is None:
                break
            done = False
            dummy_actions = np.zeros((len(obs), self.k), dtype=np.int)
            while not done:
                # put into buffer and sample for a more iid distribution
                buffer.append(obs)
                if obs.shape[0] != dummy_actions.shape[0]:
                    dummy_actions = dummy_actions[:obs.shape[0]]
                obs, _, done, _ = env.step(dummy_actions)
        del env

        log_interval = 50
        num_steps = self.hyperparameters.get("pretrain_steps")
        eval_interval = self.hyperparameters.get("pretrain_eval_steps")
        pretrain_fn = self.pretrain_state_from_batch if self.hyperparameters.get(
            "state-only-pretrain") else self.pretrain_from_batch

        eval_fn = self.pretrain_eval_fn if self.hyperparameters.get(
            "state-only-pretrain") else self.get_eval_action

        trailing_loss_values = deque(maxlen=10)
        total = min(len(buffer) // batch_size, num_steps) + 1
        with tqdm(total=total) as t:
            for i, state_batch in zip(range(total), buffer):
                state = unwrap_state(state_batch, device=self.device)
                error = pretrain_fn(state)
                trailing_loss_values.append(error)
                episode_loss = np.mean(list(trailing_loss_values))
                t.set_postfix(loss=episode_loss)
                self.log_scalar("pretrain/loss", episode_loss, global_step=i, interval=log_interval)
                if i % eval_interval == 0 and i:
                    self.post_pretrain_hook()
                    reward = self.evaluate(scope="pretrain", global_step=i, eval_fn=eval_fn)
                    if self.hyperparameters.get("state-only-pretrain"):
                        d = {
                            "rnn": self.state_agg.encoder.state_dict(),
                            "optimizer": self.state_optimizer.state_dict(),
                            "embedding": self.embedding.state_dict()
                        }
                        self.pretrain_model_saver.save_model(d, i, reward, scope="valid")
                    else:
                        self.locally_save_policy(scope="valid", reward=reward, step=i,
                                                 model_saver=self.pretrain_model_saver)
                t.update()

    def get_state_size(self):
        return self.config.hyperparameters["State"]["rnn_dim"]

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, self.info = self.environment.step(action)

        if isinstance(action, np.ndarray) and "item" in self.info:
            # put the action in the replay buffer that has the highest reward
            indices = np.array(self.info.get("item"))
            ind_mask = np.random.randint(action.shape[-1], size=indices.shape[0])
            mask = indices != -1
            ind_mask[mask] = indices[mask]
            self.action = action[np.arange(len(indices)), ind_mask]
        else:
            self.action = action[:, 0]
        self.total_episode_score_so_far += self.reward.mean()
        if self.hyperparameters["clip_rewards"]:
            self.reward = np.maximum(np.minimum(self.reward, 1.0), -1.0)

    def sample_from_action_space(self, num_items):
        subset_size = self.config.hyperparameters.get("epsilon_sample_size", self.environment.action_space.n)
        sample_subset = self.environment.get_sample_subset(subset_size, return_tensor=False)
        output_set = np.random.default_rng().choice(sample_subset, num_items, replace=False, shuffle=False)
        return output_set

    def create_state_vector(self, state):
        """
        Accumulate sequential state to one vector
        :param state:
        :return:
        """
        if isinstance(state, np.ndarray):
            state = ReturnStateTuple(*[torch.as_tensor(a, device=self.device) if a is not None else a for a in state])
        self.user_history_mask_items = state.items.cpu().t()

        curr_targets = state.targets
        state = self.state_agg(state)
        return state, curr_targets

    @abstractmethod
    def learn(self):
        pass

    @timer
    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.eval()
            self.action = self.pick_action()
            self.to_train()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            done = self.done
            if "dones" in self.info:
                done = self.info.get("dones")
                if hasattr(self.environment,
                           "max_episode_steps") and 0 < self.environment.max_episode_steps == self.episode_step_number_val:
                    # Ignore the done signal if it came from our restriction in sequence length
                    done = np.zeros_like(done)

            # Filter the last state which was not taken into account any way
            self.state = self.state[~self.last_done]
            if self.hyperparameters["batch_rl"]:
                action = self.get_batch_rl_actions().detach().numpy()
                rewards = np.ones(action.shape[0], dtype=np.float32)
                self.save_experience(experience=(self.state, action, rewards, self.next_state, done))
            self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, done))
            self.state = self.next_state
            self.global_step_number += 1
            self.last_done = done.astype(np.bool)
        self.episode_number += 1

    def get_batch_rl_actions(self):
        return unwrap_state(self.state, device="cpu").targets

    def update_linear_annealing_parameter(self, value, start_val, end_val, end_episode):
        if self.episode_number > end_episode:
            return value
        m = (end_val - start_val) / end_episode
        return m * self.episode_number + start_val

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = [self.state, self.action, self.reward, self.next_state, self.done]

        memory.add_experience(*experience)

    def post_pretrain_hook(self):
        pass

    @abstractmethod
    def _load_model(self, parameter_dict):
        """
        get the keys/values from the parameter dict for each network that we have
        """
        pass

    def load_pretrained_models(self, path=None):
        if path is None:
            path = self.model_saver.get_last_checkpoint_path()
            print("Loading model from {}".format(path))
        parameter_dict = torch.load(path, map_location=self.device)
        self.embedding.load_state_dict(parameter_dict["embedding"])

        state_dict = parameter_dict["state_agg"]
        del state_dict["context_embedding.weight"]
        self.state_agg.load_state_dict(parameter_dict["state_agg"], strict=False)
        self.state_optimizer.load_state_dict(parameter_dict["state_optimizer"])
        self.episode_number = parameter_dict["episode_number"]
        self.global_step_number = parameter_dict.get("global_step_number") or 0
        self._load_model(parameter_dict)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes_to_run
        self.durations = deque(maxlen=100)
        start = time()

        logging.info(get_output_line())
        if self.model_saver.model_exists():
            logging.info("Loading model from the log dir")
            self.load_pretrained_models()
            self.environment.continue_after_checkpoint(self.episode_number)
        elif self.hyperparameters.get("pretrain"):
            logging.info("Pre-Training the WHOLE POLICY from log-data in a supervised manner")
            if not self.pretrain_model_saver.model_exists():
                logging.info("Pretraining")
                self.pretrain()
            save_path = self.pretrain_model_saver.get_best_model_path()
            self.load_pretrained_models(save_path)
            self.post_pretrain_hook()
        elif self.hyperparameters.get("state-only-pretrain"):
            logging.info("Pre-Training the STATE from log-data in a supervised manner")
            self.pretrain()
            self.output_layer.requires_grad_(False)
            save_path = self.pretrain_model_saver.get_best_model_path()
            logging.info("loading state model from {}".format(save_path))
            rnn_dict = torch.load(save_path, map_location=self.device)
            self.state_agg.encoder.load_state_dict(rnn_dict["rnn"])
            self.state_optimizer.load_state_dict(rnn_dict["optimizer"])
            self.embedding.load_state_dict(rnn_dict["embedding"])
            self.post_pretrain_hook()

        logging.info("Starting interaction with environment in REINFORCEMENT mode")
        end_episode = num_episodes
        if self.hyperparameters.get("continue_training"):
            end_episode = self.episode_number + num_episodes
        with tqdm(initial=self.episode_number, total=end_episode) as t:
            for _ in tqdm(range(self.episode_number, end_episode)):
                ep_start = time()
                self.reset_game()
                if self.state is None:
                    break
                # Run one user batch against the agent
                self.step()
                if save_and_print_results:
                    values = self.save_and_print_result()
                    t.set_postfix(values)
                self.durations.append(time() - ep_start)

                # Save model every few episodes
                if self.config.save_model and self.episode_number % self.config.model_save_interval == 0:
                    self.locally_save_policy()

                # Run a evaluation episode on the valid data after every x episodes
                if self.episode_number % self.config.evaluation_interval == 0:
                    logging.info(get_output_line())
                    logging.info("Evaluating the agent ")
                    valid_reward = self.evaluate(scope="valid")
                    self.locally_save_policy(scope="valid", reward=valid_reward)
                t.update()

        time_taken = time() - start
        # Save final model
        if self.config.save_model and self.last_episode_returns:
            self.locally_save_policy()

        # Evaluate on the test-set with the best model on the valid-data
        best_path = self.model_saver.get_best_model_path()
        self.load_pretrained_models(best_path)
        self.config.test_environment.seed(self.config.seed)
        self.evaluate(scope="test")

        if self.tb_writer:
            self.tb_writer.flush()
        return self.last_episode_returns, self.last_episode_returns, time_taken

    def run_training_simulator(self, env_to_use):
        supervised_environment = self.environment
        self.environment = env_to_use
        self.durations = deque(maxlen=100)
        logging.info(get_output_line())
        logging.info("Starting interaction with environment in REINFORCEMENT mode")
        with tqdm(total=len(self.environment)) as t:
            while 1:
                self.reset_game()
                if self.state is None:
                    break
                # Run one user batch against the agent
                ep_start = time()
                self.step()
                self.durations.append(time() - ep_start)
                values = self.save_and_print_result(scope="simulator")
                t.set_postfix(values)
                t.update()
        if self.tb_writer:
            self.tb_writer.flush()
        self.environment = supervised_environment

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
            "update_every_n_steps"] == 0

    def get_obs(self, obs):
        items, rew, history, targets = ReturnStateTuple(*list(zip(*obs)))
        if len(obs) == 1:
            # Transforming the elements for batch-size first
            items_padded = torch.tensor(items, device=self.device, dtype=torch.long).view(-1, 1)
            if rew[0] is not None:
                rewards = torch.tensor(rew, device=self.device).view(-1, 1)
            else:
                rewards = None
            if history[0] is not None:
                history = torch.tensor(history, device=self.device)
        else:
            items = [torch.from_numpy(item) for item in items]
            items_padded = pad_sequence(items, batch_first=False).to(self.device)
            if rew[0] is not None:
                rewards = [torch.from_numpy(r) for r in rew]
                rewards = pad_sequence(rewards, batch_first=False).to(self.device)
            else:
                rewards = None

            if history[0] is not None:
                history = torch.from_numpy(np.stack(history)).to(self.device)

        return ReturnStateTuple(items_padded, rewards, history, None)

    def set_mask(self, masking: bool):
        self.masking_enabled = masking

    def create_network(self, state_size, hidden_units, output_dim, output_fn=None, **kwargs):
        return FeedforwardNetwork(input_dim=state_size,
                                  hidden_dims=hidden_units,
                                  output_dim=output_dim,
                                  output_fn=output_fn,
                                  **kwargs).to(self.device)

    def locally_save_policy(self, scope="train", reward=None, step=None, model_saver=None):
        agents = {k: v.state_dict() for k, v in self.__dict__.items() if isinstance(v, torch.nn.Module)
                  or isinstance(v, Optimizer) or isinstance(v, MultiAdam)}
        config = copy(self.config)
        config.environment = None
        config.test_environment = None
        config.valid_environment = None
        d = {
            "config": config,
            "num_envs": self.environment.num_envs,
            "episode_number": self.episode_number,
            "global_step_number": self.global_step_number
        }
        d.update(agents)

        if reward is None:
            reward = np.mean(self.last_episode_returns)
        if step is None:
            step = self.episode_number
        if model_saver is None:
            model_saver = self.model_saver
        model_saver.save_model(d, step, reward, scope=scope)

    def evaluate(self, scope="test", global_step=None, eval_fn=None, return_all_metrics=False, **kwargs):
        self.eval()
        env = self.config.test_environment if scope == "test" else self.config.valid_environment
        start = time()
        if global_step is None:
            global_step = self.episode_number
        with torch.no_grad():
            avg_return, avg_hitrate, list_div, item_div = super().evaluate(env,
                                                                           log_path=Path(self.config.tb_dir),
                                                                           tb_writer=self.tb_writer,
                                                                           global_step=global_step,
                                                                           eval_fn=eval_fn,
                                                                           scope=scope,
                                                                           **kwargs)
        end = time()
        self.to_train()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("{}/avg_return_per_user".format(scope), avg_return, global_step=global_step)
            self.tb_writer.add_scalar("{}/avg_hitrate_per_user".format(scope), avg_hitrate, global_step=global_step)
            self.tb_writer.add_scalar("{}/list_diversity".format(scope), list_div, global_step=global_step)
            self.tb_writer.add_scalar("{}/item_diversity".format(scope), item_div, global_step=global_step)
            self.tb_writer.add_scalar("{}/eval_time".format(scope), end - start, global_step=global_step)

        if return_all_metrics:
            return avg_return, avg_hitrate, list_div, item_div
        return avg_return
