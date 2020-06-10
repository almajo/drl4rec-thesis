from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import numpy as np

from environments.recommendation_env import ReturnStateTuple
from environments.simulator import ParallelSimulator


class TestSimulator(TestCase):
    def setUp(self) -> None:
        traj_file = Path("/home/test/test.csv")
        self.batch_size = 32
        self.simulator = ParallelSimulator(traj_file, num_replicas=self.batch_size, max_state_len=5,
                                           max_traj_len=20)
        self.simulator.seed(1)

    def test_reset(self):
        obs = self.simulator.reset()
        self.assertIsInstance(obs[0], ReturnStateTuple)
        self.assertEqual(len(obs), self.batch_size)

    def test_step(self):
        self.simulator.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))
        next_state, done = self.simulator.step(action)
        self.assertEqual(len(next_state), self.batch_size)

    def test_multiple_steps(self):
        self.simulator.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))
        self.simulator.step(action)
        self.simulator.step(action)
        self.simulator.step(action)
        self.simulator.step(action)
        self.simulator.step(action)
        s = self.simulator.step(action)

    def test_run_to_end(self):
        obs = self.simulator.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))
        while len(obs) > 0:
            obs, _ = self.simulator.step(action)


class TestUserLogSimulator(TestCase):
    def setUp(self) -> None:
        data_dir = Path("/home/test/")
        self.batch_size = 32
        self.simulator = ParallelSimulator(data_dir, num_replicas=self.batch_size, max_state_len=5, num_start_items=1,
                                           max_traj_len=20, logs_only=True)
        self.simulator.seed(1)

    def get_sim(self):
        return deepcopy(self.simulator)

    def test_reset(self):
        sim = self.get_sim()
        obs = sim.reset()
        self.assertIsInstance(obs[0], ReturnStateTuple)
        self.assertEqual(len(obs), self.batch_size)

    def test_step(self):
        sim = self.get_sim()
        obs = sim.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))
        next_state, done = sim.step(action)
        self.assertEqual(len(next_state), self.batch_size)

    def test_multiple_steps(self):
        sim = self.get_sim()
        sim.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))
        for _ in range(10):
            sim.step(action)

    def test_run_to_end(self):
        sim = self.get_sim()
        obs = sim.reset()
        action = np.random.randint(1, 100, size=(self.batch_size, 5))

        total = sum(e.trajectories.map(len).sum() - 1 for e in sim.envs)
        num_users = len(sim)
        counter = 0
        total_dones = 0
        while obs:
            counter += len(obs)
            obs, num_dones = sim.step(action)
            total_dones += num_dones

        self.assertEqual(total_dones, num_users)
        metrics = sim.get_metrics()
        print(metrics)

    def test_metrics(self):
        sim = self.get_sim()
        total_dones = 0
        sim.seed(1)
        obs = sim.reset()
        while obs:
            action = np.random.randint(1, 100, size=(self.batch_size, 5))
            obs, num_dones = sim.step(action)
            total_dones += num_dones
        self.assertEqual(total_dones, len(sim))
        metrics = sim.get_metrics()
        print(metrics)

        sim.hard_reset()
        sim.seed(1)
        obs = sim.reset()
        while obs:
            action = np.random.randint(1, 100, size=(self.batch_size, 5))
            obs, num_dones = sim.step(action)
        metrics_1 = sim.get_metrics()
        print(metrics_1)
        np.testing.assert_almost_equal(np.array(metrics), np.array(metrics_1))

        sim.hard_reset()
        sim.seed(1)
        obs = sim.reset()
        while obs:
            action = np.zeros((self.batch_size, 5))
            obs, num_dones = sim.step(action)
        metrics = sim.get_metrics()
        print(metrics)
