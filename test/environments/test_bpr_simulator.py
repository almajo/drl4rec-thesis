from pathlib import Path
from unittest import TestCase

import numpy as np

from environments.simulator import ParallelSimulator


class TestSimulator(TestCase):
    def setUp(self) -> None:
        data_dir = Path("/home/alex/workspace/datasets/ml/ml-1m")
        self.batch_size = 1
        self.simulator = ParallelSimulator(data_dir / "test.csv",
                                           simulation_type=data_dir / "simulator/bpr/batch-rl-test/0",
                                           num_replicas=self.batch_size,
                                           max_state_len=10,
                                           variant="bpr",
                                           reward_type="item")
        self.simulator.seed(1)

    def test_reset(self):
        obs = self.simulator.reset()
        self.assertEqual(len(obs), self.batch_size)

    def test_step(self):
        self.simulator.reset()
        action = np.random.randint(1, 3000, size=(self.batch_size, 10))
        next_state, rewards, done, info = self.simulator.step(action)
        self.assertEqual(len(next_state), self.batch_size)

    def test_multiple_steps(self):
        self.simulator.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))
        done = True
        while not done:
            next_state, rewards, done, info = self.simulator.step(action)

    def test_break_soon(self):
        episode_lens = []
        rewardsep = []
        while True:
            obs = self.simulator.reset()
            if obs is None:
                break
            done = False
            e = 0
            action = np.random.randint(1, 30, size=(len(obs), 10))
            r = 0
            while not done:
                next_state, rewards, done, info = self.simulator.step(action)
                r += rewards
                e += 1
            episode_lens.append(e)
            rewardsep.append(r)
        print("repeated")
        print(np.mean(episode_lens), np.std(episode_lens))
        print(np.mean(rewardsep), np.std(rewardsep))

    def test_run_to_end_with_much_random(self):
        episode_lens = []
        rewardsep = []
        while True:
            obs = self.simulator.reset()
            if obs is None:
                break
            done = False
            e = 0
            while not done:
                action = np.random.randint(1, 3000, size=(len(obs), 10))
                obs, rewards, done, info = self.simulator.step(action)
                rewardsep.append(rewards.mean())
                e += 1
            episode_lens.append(e)
        print("random")
        print(np.mean(episode_lens), np.std(episode_lens))
        print(np.mean(rewardsep), np.std(rewardsep))

    def test_run_to_end_tricky(self):
        episode_lens = []
        rewardsep = []

        action = np.arange(1, 101).reshape(10, 10)

        while True:
            obs = self.simulator.reset()
            if obs is None:
                break
            done = False
            e = 0
            while not done:
                a = np.expand_dims(action[e % 10], 0)
                a = np.repeat(a, len(obs), axis=0)
                obs, rewards, done, info = self.simulator.step(a)
                rewardsep.append(rewards.mean())
                e += 1
            episode_lens.append(e)
        print("tricky")
        print(np.mean(episode_lens), np.std(episode_lens))
        print(np.mean(rewardsep), np.std(rewardsep))
