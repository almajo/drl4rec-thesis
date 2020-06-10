from collections import Counter

import gym
import gym.envs
import numpy as np
import torch

from drlap.agents.Base_Agent import Base_Agent


class Player:
    def __init__(self, model_path):
        self.agent = Base_Agent.load(model_path)
        env = self.agent.environment.spec.id
        print("Playing {} with Agent {}".format(env, self.agent.agent_name))
        self.env = gym.make(env)
        self.actions = []

    def play(self, render=True, num_episodes=int(1e3)):
        self.agent.eval()

        obs = self.env.reset()
        episode = 0
        while episode < num_episodes:
            if render:
                self.env.render("human")
            obs = torch.from_numpy(obs).to(torch.float32).unsqueeze(0)
            action = self.agent.pick_action(obs)
            self.actions.append(action)
            next_obs, reward, done, info = self.env.step(action)
            if done:
                obs = self.env.reset()
                episode += 1
            else:
                obs = next_obs

        self.close()

    def close(self):
        self.env.close()

    def show_action_distribution(self, num_bins=(20, 20, 20)):
        actions = np.array(self.actions)
        min_max = np.stack([np.min(self.actions, axis=0), np.max(self.actions, axis=0)]).T

        bins = [np.linspace(m[0], m[1], n) for m, n in zip(min_max, num_bins)]

        c = Counter()
        for val in actions:
            inds = [np.digitize(d, b) for d, b in zip(val, bins)]
            c.update([tuple(inds)])

        return c
