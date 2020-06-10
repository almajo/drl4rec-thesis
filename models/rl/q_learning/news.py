from copy import deepcopy

import numpy as np
import torch

from models import ParameterizedDuellingDDQNAgent


class NewsReco(ParameterizedDuellingDDQNAgent):
    agent_name = "NEWS"

    def __init__(self, config):
        super().__init__(config)
        self.explore_network = deepcopy(self.q_network_local)
        self.explore_alpha = 0.1
        self.exploit_coefficient = 0.05

        self.exploration_indicator_mask = None

    #     if config.hyperparameters["simulator"]:
    #         self.config.hyperparameters["static_epsilon"] = 0
    #         self.exploration_strategy.constant_epsilon = 0
    #     self.is_simulator = config.hyperparameters["simulator"]
    #
    # def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
    #     if self.is_simulator:
    #         out_q_indices, exploration_indicator_mask = self.online_step(state)
    #         self.exploration_indicator_mask = exploration_indicator_mask
    #         return out_q_indices
    #     else:
    #         return super().state_to_action(state, eval_ep, top_k)

    # def conduct_action(self, action):
    #     """Conducts an action in the environment"""
    #     self.next_state, self.reward, self.done, self.info = self.environment.step(action)
    #
    #     if isinstance(action, np.ndarray) and "item" in self.info:
    #         # put the action in the replay buffer that has the highest reward
    #         indices = np.array(self.info.get("item"))
    #         ind_mask = np.random.randint(action.shape[-1], size=indices.shape[0])
    #         mask = indices != -1
    #         ind_mask[mask] = indices[mask]
    #         self.action = action[np.arange(len(indices)), ind_mask]
    #         if self.is_simulator:
    #             self.update_weights_towards_exploration(indices)
    #     else:
    #         self.action = action[:, 0]
    #     self.total_episode_score_so_far += self.reward.mean()
    #     if self.hyperparameters["clip_rewards"]:
    #         self.reward = np.maximum(np.minimum(self.reward, 1.0), -1.0)

    def online_step(self, state):
        """
        This method describe the Dueling Bandit Gradient Descent exploration method given in the original paper.
        We are first calculating the top-k list for the original network and the perturbation respectively.
        Then we directly sample the order in which we would draw them without replacement (over the whole tensor)
        Next we create a binary mask (batch-size, k) to choose the list at any timestep (but moving from first to last)

        :returns q_values, action indices, exploration mask (boolean and is 1 when the exploration network created that)

        # state = ...
        # out_q_indices, exploration_indicator_mask = online_step(state)
        # next_state, reward, done, info = env.step(out_q_indices)
        # indices = np.array(self.info.get("item"))  like in conduct action. index list of the chosen action per batch
        # minor_update:
        #   update_weights_towards_exploration(exploration_indicator_mask, indices)
        # major update:
        #   Add (s,a,r,s) to the replay-buffer and update traditionally


        """
        self._perturb_explore_network()

        state_actions = self.state_action_concat(state)

        current_qs = self.q_network_local(state, state_actions).squeeze(-1)
        indices = torch.topk(current_qs, k=self.k).indices

        explore_qs = self.explore_network(state, state_actions).squeeze(-1)
        _exp_indices = torch.topk(explore_qs, k=self.k).indices

        ranking_weight = 1. / torch.arange(1, indices.size(-1) + 1, dtype=torch.float32)
        ranking_weight = ranking_weight.unsqueeze(0).expand_as(indices)

        real_list_indices = torch.multinomial(ranking_weight, num_samples=self.k, replacement=False)
        real_list_indices = indices.gather(1, real_list_indices)

        explor_list_indices = torch.multinomial(ranking_weight, num_samples=self.k, replacement=False)
        explor_list_indices = _exp_indices.gather(1, explor_list_indices)
        #
        # out_q_indices = torch.zeros_like(indices, dtype=torch.long) - 1
        # decision_per_timestep_mask = torch.randint(2, size=indices.shape)
        #
        # bool_mask = decision_per_timestep_mask.bool()
        #
        # bool_mask_copy = bool_mask.clone().cpu().numpy()
        # output_vector = real_list_indices.clone().cpu().numpy()
        trajectories = []
        real_mask = []
        for user in range(real_list_indices.shape[0]):
            traj = []
            created_by_real = []
            real = real_list_indices[user].cpu().tolist()
            fake = explor_list_indices[user].cpu().tolist()
            for time_step in range(real_list_indices.shape[1]):
                original = np.random.randint(2)
                if original:
                    done = False
                    while not done and real:
                        element = real.pop(0)
                        if element not in traj:
                            traj.append(element)
                            done = True
                else:
                    done = False
                    while not done and fake:
                        element = fake.pop(0)
                        if element not in traj:
                            traj.append(element)
                            done = True
                created_by_real.append(original)

            trajectories.append(traj)
            real_mask.append(created_by_real)
        action = np.asarray(trajectories)
        real_mask = np.asarray(real_mask).astype(np.bool)

        # # first the real network
        # picked_indices = real_list_indices.masked_select(bool_mask)
        # out_q_indices.masked_scatter_(bool_mask, picked_indices)
        #
        # # now the exploration network
        # picked_exp_indices = explor_list_indices.masked_select(~bool_mask)
        # out_q_indices.masked_scatter_(~bool_mask, picked_exp_indices)

        return action, real_mask

    def update_weights_towards_exploration(self, indices):
        """
        Does move the current weights towards the weights of the exploration network in the following case:
        The user gave a reward for an item that the exploration policy created.
        However, if more reward was achieved with the original policy (speaking over the total batch),
        we don't do anything.
        """
        fake_indicator_mask = ~ self.exploration_indicator_mask
        if not len(indices[indices >= 0]):
            # None of the actions was a hit
            return

        batch_numbers = np.argwhere(indices >= 0).flatten()
        exploration_rewards = fake_indicator_mask[batch_numbers, indices[batch_numbers]].sum()
        real_rewards = (~fake_indicator_mask)[batch_numbers, indices[batch_numbers]].sum()
        if exploration_rewards == 0 or real_rewards > exploration_rewards:
            return

        # We got more reward with the exploration network than with the normal one -> update
        for target_param, local_param in zip(self.explore_network.parameters(), self.q_network_local.parameters()):
            local_param.data.copy_(local_param.data + self.exploit_coefficient * target_param)

    def _perturb_explore_network(self):
        # rand_uniform [-1,1]
        for target_param, local_param in zip(self.q_network_local.parameters(), self.explore_network.parameters()):
            random_factor = np.random.rand() * 2 - 1
            target_param.data.copy_(local_param.data + random_factor * self.explore_alpha * local_param.data)
