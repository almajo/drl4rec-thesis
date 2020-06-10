from typing import List

import torch
from torch import nn


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation_fn_class=torch.nn.ReLU,
                 output_fn=None, dropout=0.):
        super().__init__()
        last_dim = input_dim
        layers = []

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
            layers.append(dropout_layer)

        for h in hidden_dims:
            layers.append(torch.nn.Linear(last_dim, h))
            layers.append(activation_fn_class())
            last_dim = h

        if output_fn == "tanh":
            output_fn = torch.nn.Tanh()
        elif output_fn == "sigmoid":
            output_fn = torch.nn.Sigmoid()
        elif output_fn == "softmax":
            output_fn = torch.nn.Softmax(dim=-1)

        self.net = torch.nn.Sequential(
            *layers,
            torch.nn.Linear(last_dim, output_dim),
        )
        if output_fn:
            self.net.add_module("OutputFn", output_fn)

    def forward(self, input_):
        return self.net(input_)


class ParamDuelling(torch.nn.Module):
    # The left head is the advantage value and the right head is the Advantage-value for the given action
    def __init__(self, state_size, embedding_size, hidden_units):
        super().__init__()
        self.state_size = state_size
        self.value_net = FeedforwardNetwork(input_dim=state_size,
                                            hidden_dims=hidden_units,
                                            output_dim=1)
        self.a_net = FeedforwardNetwork(input_dim=state_size + embedding_size,
                                        hidden_dims=hidden_units,
                                        output_dim=1)

    def forward(self, states, state_actions):
        # state dim (bs, state_dim), state_actions dim (bs, n, state_dim+action_dim)
        # when calculating the value for targets, we need a sample of more values in order to calculate a sample_mean
        values = self.value_net(states)

        advantages = self.a_net(state_actions).squeeze(-1)
        mean_advantage = advantages.mean(dim=1, keepdim=True)
        centered_a_values = advantages - mean_advantage

        q_values = values + centered_a_values

        return q_values.unsqueeze(-1)

    def predict(self, state_actions):
        return self.a_net(state_actions)


class DuelingPolicy(torch.nn.Module):
    # The left head is the advantage value and the right head is the Advantage-value for the given action
    def __init__(self, state_size, hidden_units, output_dim):
        super().__init__()
        self.state_size = state_size
        self.value_net = FeedforwardNetwork(input_dim=state_size,
                                            hidden_dims=hidden_units,
                                            output_dim=1)
        self.a_net = FeedforwardNetwork(input_dim=state_size,
                                        hidden_dims=hidden_units,
                                        output_dim=output_dim)

    def forward(self, states):
        values = self.value_net(states)
        advantages = self.a_net(states)
        mean_advantage = advantages.mean(dim=1, keepdim=True)
        centered_a_values = advantages - mean_advantage

        q_values = values + centered_a_values
        return q_values

    def predict(self, states):
        return self.a_net(states)


class CorrectionReinforceMainPolicy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = self.hidden_layer(state)
        x = torch.relu(x)
        logits = self.output_layer(x)
        return logits, x

    def predict(self, state):
        logits = self.forward(state)[0]
        return logits.softmax(dim=-1)
