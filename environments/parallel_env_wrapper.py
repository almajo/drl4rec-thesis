from copy import copy, deepcopy

import numpy as np
import torch
from gym import Env


class ParallelEnvWrapper(Env):
    def __init__(self, env, num_replicas):
        """
        A wrapper around an env that plays _num_replicas_ games simultaneously per _step_.
        :param env: Test Environment object that will be wrapped
        :param num_replicas: Number of parallel instances of the env, equivalent to batch_size
        """
        self.env = env
        self.id = env.id
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.max_episode_steps = env.max_episode_steps
        data = env.data
        num_replicas = min(len(data), num_replicas)
        self.num_envs = num_replicas
        data_split = map(list, np.array_split(data, num_replicas))

        envs = []
        for x in data_split:
            test_env = copy(env)
            test_env.data = x
            test_env.num_repeats = env.num_repeats
            envs.append(test_env)
        self.envs = np.array(envs, dtype=np.object)
        self.env_alive_mask = np.ones(len(envs), dtype=np.int)
        self.done_mask = np.zeros(len(envs), dtype=np.int)
        self.next_round_actions = np.arange(len(envs))
        self.rng = np.random.default_rng(0)

    @property
    def base_env(self):
        return deepcopy(self.env)

    def step(self, action: np.ndarray):
        """
        Do a step in all environments.
        :param action: nd.array (-1, k) where k is the metrics cutoff: First dimension is first num_replicas but
        gets smaller as envs exhaust
        :return: (observations, num_users_done) where obs is list of obs-tuples
        """
        indices = np.flatnonzero(self.env_alive_mask & ~self.done_mask)

        observations = []
        rewards = []
        item_index = []
        dones = []

        for idx, env_i in zip(self.next_round_actions, indices):
            env = self.envs[env_i]
            sel_action = action[idx]
            obs, r, d, i = env.step(sel_action)
            observations.append(obs)
            rewards.append(r)
            item_index.append(i.get("item"))
            dones.append(int(d))
            if d:
                self.done_mask[env_i] = 1
        self.next_round_actions = np.flatnonzero(1 - np.array(dones))
        info = {
            "item": np.array(item_index, dtype=np.long),
            "dones": np.array(dones, dtype=np.long),
            "indices": indices
        }
        total_done = (self.done_mask == self.env_alive_mask).all()
        return np.array(observations, dtype=np.object), np.array(rewards), total_done, info

    @property
    def active_environments(self):
        return np.flatnonzero(self.env_alive_mask & ~self.done_mask)

    def hard_reset(self):
        self.env_alive_mask = np.ones(len(self.envs), dtype=np.int)
        for e in self.envs:
            e.hard_reset()

    def reset(self):
        self.done_mask = np.zeros(len(self.envs), dtype=np.int)
        self.next_round_actions = np.arange(len(self.envs))

        envs, indices = self.get_alive_envs()
        start_states = []
        for i, e in zip(indices, envs):
            state = e.reset()
            if state is not None:
                start_states.append(state)
            else:
                self.env_alive_mask[i] = 0

        if len(start_states) == 0:
            return None

        return np.array(start_states)

    def get_alive_envs(self):
        indices = self.env_alive_mask.nonzero()[0]
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

    def render(self, mode='human'):
        pass

    def __len__(self):
        return sum(map(len, self.envs))
