import numpy as np
import torch

from models.RecommendationAgent import RecommendationAgent
from util.helpers import unwrap_state
from util.sequential_replay import SequentialReplay


class GRU(RecommendationAgent):
    agent_name = "GRU"
    name = "GRU"

    def __init__(self, config):
        super().__init__(config)
        self.agent = self.get_nn()
        self.optimizer = torch.optim.Adam(self.agent.parameters(),
                                          lr=self.hyperparameters["learning_rate"],
                                          weight_decay=self.hyperparameters["weight_decay"])
        self.loss = torch.nn.CrossEntropyLoss()
        self.memory = SequentialReplay(config, config.seed, self.device)

        self.loss_vals = []
        self.last_episode = 0

    def get_nn(self):
        return self.create_network(self.state_size,
                                   self.hyperparameters["linear_hidden_units"],
                                   self.environment.action_space.n).to(self.device)

    def learn(self):
        """Runs a learning iteration for the network"""
        states, actions, rewards, _, _ = self.memory.sample()
        positive_rewards = (rewards.squeeze() > 0).nonzero()
        if positive_rewards.nelement() > 0:
            indices = positive_rewards.squeeze(-1)
            state, _ = self.create_state_vector(states)
            state = state[indices]
            logits = self.agent(state)
            error = self.loss(logits, actions.squeeze()[indices])
            self.take_optimisation_step([self.state_optimizer, self.optimizer], self.agent, error,
                                        self.hyperparameters.get("gradient_clipping_norm"))

        if self.loss_vals and self.last_episode != self.episode_number:
            self.log_scalar("train/loss", np.mean(self.loss_vals))
            self.loss_vals = []
            self.last_episode = self.episode_number

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
            action = self.state_to_action(state, eval_ep, top_k)
        return action

    def state_to_action(self, state, eval_ep, top_k) -> np.ndarray:
        logits = self.agent(state)
        probs = torch.softmax(logits, dim=-1)
        probs[..., 0] = 0
        samples = torch.multinomial(probs + 1e-10, top_k)

        return samples.cpu().numpy()

    def _supervised_learning_from_batch(self, state: torch.Tensor, targets: torch.Tensor) -> float:
        logits = self.agent(state)
        error = self.loss(logits, targets)

        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()
        return error.item()

    def _load_model(self, parameter_dict):
        self.agent.load_state_dict(parameter_dict["agent"])
        self.optimizer.load_state_dict(parameter_dict["optimizer"])

    def get_eval_action(self, obs, k):
        state, _ = self.create_state_vector(obs)
        logits = self.agent(state)
        probs = torch.softmax(logits, dim=-1)
        if self.masking_enabled:
            probs.scatter_(1, self.user_history_mask_items.to(probs.device), float("-inf"))
        probs[..., 0] = float("-inf")
        top_k = torch.topk(probs, k)
        return top_k.indices.detach().cpu().numpy()
