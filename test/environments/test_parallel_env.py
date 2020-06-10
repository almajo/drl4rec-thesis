from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import numpy as np
from tqdm import tqdm

from environments import RecommendEnv
from environments.parallel_env_wrapper import ParallelEnvWrapper


class TestParallelTestEnv(TestCase):
    def setUp(self) -> None:
        data_dir = Path("/home/alex/workspace/datasets/ml/ml-1m")
        top_k_actions = 10
        max_episode_len = 0
        max_state_len = 10
        num_parallel_envs = 32
        self.num_replicas = num_parallel_envs
        env = RecommendEnv(debug=True).initialize(data_dir / "train_split.csv",
                                                  num_repeats=1,
                                                  sample_k=top_k_actions,
                                                  max_episode_len=max_episode_len,
                                                  max_state_len=max_state_len,
                                                  )
        env = ParallelEnvWrapper(env, num_parallel_envs)
        self.env = env

    def test_step(self):
        env = deepcopy(self.env)
        env.reset()
        obs = env.step(np.zeros((self.num_replicas, 20)))

    def test_reset(self):
        env = deepcopy(self.env)
        obs = env.reset()
        self.assertEqual(len(obs), self.num_replicas)

    def test_no_env_skip(self):
        env = deepcopy(self.env)
        num_episodes = 10_000
        for _ in tqdm(range(num_episodes)):
            obs = env.reset()
            if obs is None:
                break
            step = 0
            while True:
                targets = np.stack([o[-1] for o in obs])
                obs, reward, done, info = env.step(targets)
                self.assertTrue((reward == 1).all())
                step += 1
                if done:
                    break
