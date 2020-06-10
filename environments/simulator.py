import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from gym import Env
from gym.spaces import Discrete

# from environments.gru_simulator import GRUSimulator
from drlap.utilities.data_structures.Config import Config
from environments.recommendation_env import ReturnStateTuple
from environments.reward_functions import target_in_action_reward
from models import GRU
from util.helpers import flatten
from util.in_out import read_info_file

"""
Simulator for test purposes
The simulator starts with the first k items from some users trajectory
The Reco-Agent then returns action(s) to the simulator.
The Simulator tosses a coin and accepts the action with respect to the previously-generated user-item-matrix

IF the action is accepted:
    we leave the original trajectory recorded in the test-data set and continue from the new one.
ELSE
    we jump back to the trajectory (Which mimics clicking on another object than we recommended)
    
Simultaneously we incorporate a "getting bored"-probability that a user doesn't get anything interesting and thus 
leaves the platform (end of trajectory)

The simulator only treats the most likely item in the k-provided ones
"""


class Simulator(Env):

    def __init__(self, trajectories, max_state_len=10, max_traj_len=50, reward_type="item", **kwargs):
        super().__init__()
        self.trajectories = trajectories
        self.max_state_len = max_state_len
        self.max_episode_steps = max_traj_len
        self.top_k = None
        self.reward_type = reward_type
        self.hard_reset()

        self.default_leave_prob = kwargs.get("default_leave_prob", 0.1)
        self.other_item_leave_prob = kwargs.get("other_item_leave_prob", 0.1)
        self.repetition_factor = kwargs.get("repetition_factor", 0.2)

        self.current_state = None

    @abstractmethod
    def _receive_reward(self, action: np.ndarray) -> np.ndarray:
        """
        Returns a numpy array of same shape as action, that has the respective reward in it
        """
        return action

    @abstractmethod
    def _accept_item(self, actions, rewards) -> int:
        """
        Returns the item-id from actions that was accepted. If none is to be accepted, return value is -1
        """
        return -1

    @abstractmethod
    def _interrupt_trajectory(self, leave_probability) -> bool:
        """
        Returns True if we are interrupting the session based on the leave_probability
        """
        return False

    def step(self, action: np.ndarray):
        self.time_step += 1

        reward_list = self._receive_reward(action)
        leave_prob = self.default_leave_prob

        next_click = self._accept_item(action, reward_list)
        if next_click == -1:
            # We do not accept the action, but click on something else, i.e. go back to user log
            next_click = self.current_user_real_trajectory[self.real_next_index]
            self.real_next_index += 1
            # while self.real_next_index < len(self.current_user_real_trajectory):
            #     next_click = self.current_user_real_trajectory[self.real_next_index]
            #     self.real_next_index += 1
            #     if next_click not in self.current_trajectory:
            #         break
            leave_prob = self.other_item_leave_prob

        if self.last_recommendation_actions:
            last = np.in1d(action, self.last_recommendation_actions[-1])
            percentile_repeats = last.sum() / len(last)
            # Modelling the impatience and annoyance of repetitive recommendations
            leave_prob += self.repetition_factor * percentile_repeats

        self.current_trajectory = np.concatenate([self.current_trajectory, np.array([next_click])])

        interrupt_session = self._interrupt_trajectory(leave_prob)
        done = interrupt_session or \
               self.real_next_index == len(self.current_user_real_trajectory) or \
               0 < self.max_episode_steps == self.time_step

        self.last_recommendation_actions.append(action)

        total_reward = 0
        if self.reward_type == "item":
            total_reward = float(next_click in action)
            # if next_click in action:
            #     total_reward = reward_list[action == next_click].item()
        else:
            total_reward = max(reward_list.sum(), 0)

        if interrupt_session:
            total_reward = 0

        self.episode_rewards.append(total_reward)

        if done:
            self.store_metrics()

        info = {"item": -1 if next_click not in action else np.flatnonzero(next_click == action)[0]}

        self.current_state = ReturnStateTuple(self.current_trajectory[-self.max_state_len:], None, None, None)
        return self.current_state, total_reward, done, info

    def store_metrics(self):
        if self.last_recommendation_actions:
            # HERE ARE THE METRICS DEFINED
            accuracy = sum(self.episode_rewards)
            self.all_returns.append(accuracy)
            self.hit_rates.append(accuracy / len(self.episode_rewards))

            sorted_recs = np.sort(self.last_recommendation_actions, axis=-1)
            self.all_unique_items.update(np.unique(sorted_recs.ravel()))
            self.all_unique_item_lists.update(map(lambda x: tuple(np.ravel(x)), sorted_recs))
            self.num_actions_recommended += len(sorted_recs)

    def session_start_index(self):
        return 0

    def reset(self):
        self.user_index += 1
        if self.user_index == len(self.trajectories):
            # end of this simulator
            return None

        self.current_user_real_trajectory = np.array(self.trajectories.loc[self.user_index])

        time_start = self.session_start_index()
        self.current_trajectory = np.array([self.current_user_real_trajectory[time_start]])
        self.last_recommendation_actions = []
        self.episode_rewards = []
        self.real_next_index = time_start + 1

        self.current_state = ReturnStateTuple(self.current_trajectory, None, None, None)
        return self.current_state

    def get_metrics(self):
        if self.last_recommendation_actions:
            self.top_k = len(self.last_recommendation_actions[-1])
        return self.all_returns, self.hit_rates, self.all_unique_item_lists, self.all_unique_items, self.num_actions_recommended

    def hard_reset(self):
        self.user_index = -1
        self.time_step = 0
        self.current_trajectory = None
        self.current_user_real_trajectory = None
        self.real_next_index = 1

        self.last_recommendation_actions = []
        self.episode_rewards = []
        self.hit_rates = []
        self.all_returns = []
        self.all_unique_item_lists = set()
        self.all_unique_items = set()
        self.num_actions_recommended = 0

    def seed(self, seed=None):
        np.random.seed(seed)

    def set_user_index(self, episode_number):
        self.user_index = episode_number % len(self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def render(self, mode='human'):
        pass


class UserLogFollowingSimulator(Simulator):

    # DO NOT TOUCH THIS WAS USED FOR EVALUATIONS ALREADY
    def __init__(self, trajectories, max_state_len, **kwargs):
        super().__init__(trajectories, max_state_len, **kwargs)

    def _receive_reward(self, action: np.ndarray):
        pass

    def _accept_item(self, actions, rewards) -> int:
        pass

    def _interrupt_trajectory(self, leave_probability) -> bool:
        pass

    def step(self, action: np.ndarray):
        self.time_step += 1

        next_user_action = self.current_user_real_trajectory[self.time_step]
        reward, index_in_action_list = target_in_action_reward(action, next_user_action)

        self.current_trajectory = np.concatenate([self.current_trajectory, np.array([next_user_action])])

        done = self.time_step + 1 == len(self.current_user_real_trajectory) or \
               self.max_episode_steps == self.time_step

        self.last_recommendation_actions.append(action)
        self.episode_rewards.append(reward)

        if done:
            self.store_metrics()

        return ReturnStateTuple(self.current_trajectory[-self.max_state_len:], None, None, None), reward, done, \
               {"item": index_in_action_list}

    def reset(self):
        self.user_index += 1

        if self.user_index == len(self.trajectories):
            # end of this simulator
            return None
        self.current_user_real_trajectory = np.array(self.trajectories.loc[self.user_index])
        self.current_trajectory = np.array([self.current_user_real_trajectory[0]])

        self.last_recommendation_actions = []
        self.episode_rewards = []
        self.time_step = 0
        return ReturnStateTuple(self.current_trajectory, None, None, None)


class BPRSimulator(Simulator):

    def __init__(self, trajectories, rewards, **kwargs):
        super().__init__(trajectories, **kwargs)
        self.rewards = rewards
        self.num_items = rewards.shape[-1]
        self.other_item_reward = kwargs.get("other_item_reward", 1.)

        self.all_mean_rewards = []
        self.per_trajectory_mean_list_reward = []

    def _get_current_user_rewards(self):
        return self.rewards[self.user_index]

    def _receive_reward(self, action: np.ndarray):
        user_row = self._get_current_user_rewards()
        rewards_for_actions = user_row[action]
        return rewards_for_actions

    def session_start_index(self):
        return np.random.randint(len(self.current_user_real_trajectory) - 2)

    def _accept_item(self, actions, rewards) -> int:
        list_reward = np.mean(rewards)
        pos_rewards = rewards > 0
        rewards = rewards[pos_rewards]
        actions = actions[pos_rewards]
        if len(rewards) == 0:
            return -1
        self.per_trajectory_mean_list_reward.append(list_reward)
        rewards = rewards * np.abs(list_reward)

        other_item_reward = np.array([max(self.other_item_reward, list_reward)])
        rewards = np.concatenate([other_item_reward, rewards])
        actions = np.concatenate([np.array([-1], dtype=np.long), actions])
        exps = np.exp(rewards)
        probs = exps / exps.sum()
        chosen_action = np.random.choice(actions, p=probs)
        return chosen_action

    def _interrupt_trajectory(self, leave_probability) -> bool:
        return np.random.rand() <= leave_probability

    def sample(self, num_samples):
        return np.random.choice(np.arange(1, self.num_items), size=num_samples, replace=False)

    def reset(self):
        if self.per_trajectory_mean_list_reward:
            self.all_mean_rewards.append(np.mean(self.per_trajectory_mean_list_reward))
            self.per_trajectory_mean_list_reward = []
        return super().reset()


class ParallelSimulator:
    def __init__(self, trajectory_file: Path, simulation_type="test", num_replicas=16, max_state_len=10,
                 max_traj_len=50, variant="logs", iteration_level=2, **kwargs):
        """
        :param num_replicas:
        :param max_state_len:
        :param max_traj_len:
        :param variant: One of [logs, bpr, gru]
        """
        self.id = trajectory_file.parent.name
        trajectories = pd.read_csv(trajectory_file, usecols=["userId", "movieId", "timestamp"])
        trajectories = trajectories.sort_values(by=["userId", "timestamp"]).groupby("userId").agg(list)
        trajectories = trajectories.movieId.squeeze()
        trajectories.reset_index(drop=True, inplace=True)

        data_split = np.array_split(trajectories, num_replicas)
        self.max_state_len = max_state_len
        self.num_items = int(read_info_file(trajectory_file.parent).get("num_items"))
        self.action_space = Discrete(self.num_items + 1)
        self.num_envs = num_replicas
        self.infinite = simulation_type == "train"
        envs = []
        if variant == "logs":
            for traj in data_split:
                traj = traj.reset_index(drop=True)
                k = UserLogFollowingSimulator(traj,
                                              max_traj_len=max_traj_len,
                                              max_state_len=max_state_len)
                envs.append(k)
        elif variant == "bpr":
            print("Using the {} simulator".format(simulation_type))
            sim_dir = trajectory_file.parent / "simulator/bpr/{}-simulator/{}".format(simulation_type, iteration_level)
            rewards = torch.load(sim_dir / "centered_scores.pkl").detach().cpu().numpy()
            i = 0
            for traj in data_split:
                rew = rewards[i:i + len(traj)]
                assert len(rew) == len(traj)
                i += len(rew)
                traj = traj.reset_index(drop=True)
                k = BPRSimulator(traj,
                                 rew,
                                 max_state_len=max_state_len,
                                 max_traj_len=max_traj_len,
                                 **kwargs)
                envs.append(k)
        elif variant == "gru":
            for traj in data_split:
                traj = traj.reset_index(drop=True)
                model_dir = kwargs.get("model_dir")
                k = GRUSimulator(traj, model_dir, max_traj_len=max_traj_len, max_state_len=max_state_len)
                envs.append(k)
        else:
            raise NotImplementedError
        self.envs = np.array(envs, dtype=np.object)
        self.env_alive_mask = np.ones(len(envs), dtype=np.bool)
        self.done_mask = np.zeros(len(envs), dtype=np.bool)
        self.last_dones = np.zeros_like(self.done_mask, dtype=np.bool)
        self.rng = np.random.default_rng(0)

    def seed(self, seed=0):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def step(self, action: np.ndarray):
        """
        Do a step in all environments.
        :param action: nd.array (-1, k) where k is the metrics cutoff: First dimension is first num_replicas but
        gets smaller as envs exhaust
        :return: (observations, num_users_done) where obs is list of obs-tuples
        """
        indices = np.flatnonzero(self.env_alive_mask & ~self.done_mask)
        actions_to_take = np.flatnonzero(~self.last_dones)
        observations = []
        rewards = []
        info = []
        dones = []
        for idx, env_i in zip(actions_to_take, indices):
            env = self.envs[env_i]
            sel_action = action[idx]
            obs, r, d, i = env.step(sel_action)
            observations.append(obs)
            rewards.append(r)
            info.append(i)
            dones.append(int(d))
            if d:
                self.done_mask[env_i] = True

        info = {k: [i[k] for i in info] for k in info[0].keys()}
        info["dones"] = np.array(dones, dtype=np.long)
        info["indices"] = indices
        total_done = (self.done_mask == self.env_alive_mask).all()
        self.last_dones = np.asarray(dones, dtype=np.bool)
        return np.array(observations, dtype=np.object), np.array(rewards), total_done, info

    def get_metrics(self):
        from functools import reduce
        metrics = [e.get_metrics() for e in self.envs]
        total_return, hit_rates, list_diversity, item_diversity, num_actions = list(zip(*metrics))
        item_diversity = len(reduce(lambda a1, a2: a1 | a2, item_diversity))
        list_diversity = len(reduce(lambda a1, a2: a1 | a2, list_diversity))
        num_actions = sum(num_actions)

        mean_hitrate = np.mean(flatten(hit_rates))
        mean_return = np.mean(flatten(total_return))
        mean_item_div = item_diversity / self.num_items
        mean_list_div = list_diversity / num_actions

        return mean_return, mean_hitrate, mean_list_div, mean_item_div

    def hard_reset(self):
        self.env_alive_mask = np.ones(len(self.envs), dtype=np.bool)
        for e in self.envs:
            e.hard_reset()

    def reset(self):
        self.done_mask = np.zeros(len(self.envs), dtype=np.bool)
        self.last_dones = np.zeros_like(self.done_mask)

        envs, indices = self.get_alive_envs()
        start_states = []
        for i, e in zip(indices, envs):
            state = e.reset()
            if state is None:
                if self.infinite:
                    e.hard_reset()
                    state = e.reset()
                else:
                    self.env_alive_mask[i] = False
                    continue
            start_states.append(state)

        if len(start_states) == 0:
            return None

        return np.array(start_states)

    def get_alive_envs(self):
        indices = np.flatnonzero(self.env_alive_mask)
        return self.envs[indices], indices

    def get_sample_subset(self, num_samples: int, return_tensor=True):
        env_samples = np.stack([e.sample(num_samples) for e in self.envs])
        if return_tensor:
            return torch.from_numpy(env_samples)
        return env_samples

    def sample_unrestricted(self, *shape):
        n = self.action_space.n
        if np.prod(shape) > n:
            samples = np.stack(
                [self.rng.choice(n, size=shape[-1], replace=False, shuffle=False) for _ in range(shape[0])])
        else:
            samples = self.rng.choice(self.action_space.n, size=shape,
                                      replace=False, shuffle=False)
        return torch.from_numpy(samples)

    def continue_after_checkpoint(self, episode_number):
        for env in self.envs:
            env.set_user_index(episode_number)

    def __len__(self):
        return sum(map(len, self.envs))


def get_base_config(trajectory_file, experiment_dir, seed):
    top_k_actions = 10

    max_state_len = 10
    test_max_len = 100
    test_env = ParallelSimulator(trajectory_file, simulation_type="test", num_replicas=1,
                                 max_state_len=max_state_len,
                                 max_traj_len=test_max_len, variant="logs")

    config = Config()
    config.seed = seed
    config.environment = test_env
    config.test_environment = test_env
    config.valid_environment = None
    config.num_episodes_to_run = 1
    config.model_save_interval = 1000000000  # save model every n epochs, overwriting oneself
    config.evaluation_interval = 5000000000
    config.file_to_save_model = str(experiment_dir / "model.pkl")
    config.tb_dir = str(experiment_dir)
    config.tb_log_interval = 10
    config.show_solution_score = False
    config.use_GPU = torch.cuda.is_available()
    config.overwrite_existing_results_file = True
    config.randomise_random_seed = False
    config.save_model = False
    config.render_from = -1
    config.metrics_k = top_k_actions
    config.use_tb = False
    return config


class GRUSimulator(Simulator):

    def __init__(self, trajectories, model_dir=None, **kwargs):
        super().__init__(trajectories, **kwargs)

        DATA_PATH = Path(os.getenv("DATA_PATH", "/home/stud/grimmalex/datasets/"))
        OUT_PATH = Path(os.getenv("OUT_PATH", "/home/stud/grimmalex/thesis/output/"))
        if model_dir is None:
            model_dir = OUT_PATH / "gru-sim/first/ml-1m/gru/2"
            print("model dir is {}".format(model_dir))
        data, agent, seed = str(model_dir).split("/")[-3:]
        if "ml" in data:
            data = "ml/{}".format(data)
        data_dir = DATA_PATH / data

        trajectory_file = data_dir / "test.csv"

        config = get_base_config(trajectory_file, Path(model_dir), 1)
        with open(model_dir / "hyperparameters.yaml", "r") as f:
            hyperparameters = yaml.load(f, yaml.Loader)
        main_key = list(hyperparameters.keys())[0]
        config.hyperparameters = hyperparameters[main_key]

        w2v_path = config.hyperparameters["Embedding"]["w2v_context_path"]
        if str(DATA_PATH) not in w2v_path:
            end_path = str(w2v_path).split("datasets/")[-1]
            w2v_path = DATA_PATH / end_path
            config.hyperparameters["Embedding"]["w2v_context_path"] = w2v_path
        agent = GRU(config)
        path = agent.model_saver.get_last_checkpoint_path()
        agent.load_pretrained_models(path)
        self.agent = agent

        self.reward_type = "list"

    def forward(self, action):
        with torch.no_grad():
            state = self.current_state
            state = ReturnStateTuple(torch.as_tensor(state.items, device=self.agent.device).unsqueeze(1), None, None,
                                     None)
            state = self.agent.state_agg(state)
            logits = self.agent.agent(state)
            reward = torch.softmax(logits, dim=-1).cpu().numpy()

        action_rewards = reward[..., action]
        return action_rewards.flatten()

    def _receive_reward(self, action: np.ndarray) -> np.ndarray:
        return self.forward(action)

    def _accept_item(self, actions, rewards) -> int:
        sum_ = rewards.sum(axis=-1)
        if sum_ >= 0.05:
            action = np.random.choice(actions, p=rewards / rewards.sum())
            return action
        return -1

    def _interrupt_trajectory(self, leave_probability) -> bool:
        return np.random.rand() <= leave_probability


def test_parallel():
    data_path = Path("/home/stud/grimmalex/datasets/retailrocket/simulator")
    sim = ParallelSimulator(data_path, variant="logs")

    obs = sim.reset()
    obs = sim.step(np.zeros((len(obs), 5)))


if __name__ == '__main__':
    test_parallel()
