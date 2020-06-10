import logging
from collections import deque
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete

from environments.reward_functions import target_in_action_reward
from util.in_out import read_info_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ReturnStateTuple = namedtuple("ReturnStateTuple", "items rewards history targets")


class OneUpActionSpace(Discrete):
    # Do not sample the zero (embedding)
    def __init__(self, n, k=1):
        super().__init__(n)
        self.k = k

    def sample(self, batch_size=1, k=None):
        if not k:
            k = self.k
        return np.random.default_rng().choice(np.arange(1, self.n), size=(batch_size, k), replace=False, shuffle=False)


ENV_FINISHED_TARGET = 0


class RecommendEnv(Env):

    def __init__(self, debug=False):
        self.id = "unnamed"
        self.max_episode_steps = 0
        self.data = []

        self.finished_repeats = 0
        self.user_index = 0
        self.time_step = 0
        self.num_repeats = 1
        self.max_state_len = 0
        self.num_start_state_items = 1

        self.full_trajectory = deque()

        self.finished = False
        self.sample_k = 20

        self.shuffle = False
        self.debug = debug

        self.rng = np.random.default_rng(1)

    def initialize(self, trajectory_file: Path,
                   num_repeats=1,
                   max_episode_len=0,
                   max_state_len=10,
                   num_start_state_items=1,
                   sample_k=20):
        data_dir = trajectory_file.parent
        # if str(trajectory_file).endswith("test.csv"):
        #     data_dir = data_dir.parent

        self.id = data_dir.name
        logger.info("Loading the dataset")
        num_rows = 10000 if self.debug else None
        data = pd.read_csv(trajectory_file, nrows=num_rows, usecols=["userId", "movieId"], na_filter=False)
        self.data = data.groupby("userId").agg(list).squeeze().tolist()

        data_info = read_info_file(data_dir)
        largest_id = int(data_info.get("num_items"))
        logger.info("Number of items: {}".format(largest_id))

        # Action 0 is padding
        self.action_space = OneUpActionSpace(largest_id + 1, k=sample_k)
        self.reward_range = (0, 1)
        self.max_episode_steps = max_episode_len
        self.num_repeats = num_repeats
        self.num_start_state_items = num_start_state_items
        self.sample_k = sample_k
        self.max_state_len = max_state_len

        logger.info("Finished loading data")
        logger.info("Number of users: {}".format(len(self.data)))

        return self

    def step(self, action: np.ndarray):
        if action.size > 1:
            action = np.squeeze(action)

        target_id = self._get_target_id()
        reward, index_in_action_list = target_in_action_reward(action, target_id)
        self.time_step += 1

        output_dict = {"item": index_in_action_list}

        if self.done() or (self.max_episode_steps and self.max_episode_steps == self.time_step):
            next_dummy_state = self.full_trajectory[:self.time_step + 1]
            if 0 < self.max_state_len < len(next_dummy_state):
                next_dummy_state = next_dummy_state[-self.max_state_len:]
            return ReturnStateTuple(next_dummy_state, None, None,
                                    np.array([ENV_FINISHED_TARGET])), reward, True, output_dict

        current_items, next_target = self._get_current_state_and_target()

        return ReturnStateTuple(current_items, None, None, next_target), reward, False, output_dict

    def done(self):
        return (self.time_step + 1) == len(self.full_trajectory)

    def sample(self, num_samples: int):
        """
        Returns a np array of num_samples+1 where the 1 is the correct target for a limited exploration space
        """
        assert num_samples < self.action_space.n
        uniform_samples = self.rng.choice(self.action_space.n, size=num_samples, replace=False, shuffle=False)
        if (self.time_step + 1) == len(self.full_trajectory):
            # It does not matter because this environment is done any way
            return uniform_samples

        target = self._get_target_id()
        if target in uniform_samples:
            return uniform_samples
        target = np.array([target], dtype=np.int64)
        samples = np.concatenate([uniform_samples[1:], target])  # Only use num_samples-1 to return num_sample items
        return samples

    def _get_target_id(self):
        return self.full_trajectory[self.time_step + 1]

    def _get_current_state_and_target(self):
        current_items = self.full_trajectory[:self.time_step + 1]
        target = np.array([self._get_target_id()], dtype=np.long)
        # clip state history
        if 0 < self.max_state_len < len(current_items):
            current_items = current_items[-self.max_state_len:]
        return current_items, target

    def reset(self):
        if self.user_index == len(self.data):
            # Reset user_index (start dataset iteration from scratch)
            self.finished_repeats += 1
            self.user_index = 0
            # shuffle data
            if self.shuffle:
                self.data = self.data.sample(frac=1, random_state=666 + self.finished_repeats)
                self.data.reset_index(inplace=True, drop=True)
        if self.finished_repeats == self.num_repeats:
            # Num iterations finished
            self.finished = True
            return None
        self.full_trajectory = np.array(self.data[self.user_index], dtype=np.long)

        # reset
        self.time_step = 0

        self.user_index += 1
        start_items, target = self._get_current_state_and_target()
        return ReturnStateTuple(start_items, None, None, target)

    def hard_reset(self):
        self.user_index = 0
        self.finished_repeats = 0
        self.finished = False

    def render(self, mode='human'):
        pass

    def set_user_index(self, episode_number):
        self.user_index = episode_number % len(self.data)

    def __len__(self):
        return len(self.data)


class RecommendationEnvWithRecHistory(RecommendEnv):
    def __init__(self):
        super().__init__()
        self.last_recommendations = list()

    def reset(self):
        obs = super().reset()
        if obs is None:
            return None
        start_items, start_rewards, _, target = obs
        self.last_recommendations = list()
        last_items = np.zeros(self.sample_k, dtype=np.long)
        return ReturnStateTuple(start_items, start_rewards, last_items, target)

    def step(self, action: np.ndarray):
        (current_items, current_rewards, _, next_target), reward, done, info = super().step(action)
        return ReturnStateTuple(current_items, current_rewards, action, next_target), reward, done, info
