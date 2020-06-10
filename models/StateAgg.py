import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from environments.recommendation_env import ReturnStateTuple
from util.custom_embeddings import CustomEmbedding
from util.helpers import get_seq_len


class RNNStateAgg(nn.Module):
    def __init__(self, embedding: CustomEmbedding, state_config, one_hot_dim=None, reward_range=None,
                 with_rewards=False, content=False, *args, **kwargs):
        super().__init__()
        if with_rewards:
            assert reward_range and one_hot_dim
        self.use_rewards_concat = with_rewards
        self.embedding_dim = embedding.embedding_dim
        self.reward_one_hot_dim = one_hot_dim
        self.reward_range = reward_range
        self.context_embedding = embedding
        self.content = content
        self.state_size = state_config["rnn_dim"]

        input_size = self.embedding_dim + one_hot_dim if with_rewards else self.embedding_dim
        self.encoder = nn.GRU(input_size=input_size, hidden_size=state_config["rnn_dim"], batch_first=False)

    def parameters(self, recurse: bool = ...):
        return self.encoder.parameters()

    def check_input(self, *args):
        if args[0].dim() == 1:
            for i in args:
                i.unsqueeze_(1)

    def forward(self, state: ReturnStateTuple):
        """
        :param state: ReturnStateTuple for different versions
        :param seq_lens:
        :return:
        """
        self.check_input(state.items, state.rewards)
        x = self.context_embedding(state.items)

        if x.size(1) > 1:
            seq_lens = get_seq_len(state)
            x = pack_padded_sequence(x, seq_lens, batch_first=False, enforce_sorted=False)

        _, h = self.encoder(x)

        state = h.squeeze(0)
        return state


class TransformerRNNStateAgg(RNNStateAgg):
    def __init__(self, embedding: CustomEmbedding, output_size: int, nheads=4, n_layers=2,
                 dropout=0.1, **kwargs):
        super().__init__(embedding, output_size, **kwargs)
        layer = nn.TransformerEncoderLayer(embedding.embedding_dim, nheads,
                                           dim_feedforward=output_size, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.state_size = output_size

    def forward(self, state):
        self.check_input(state.items, state.rewards)

        x = self.context_embedding(state.items)

        # Only make this reward stuff if we use it
        if self.use_rewards_concat:
            one_hot_rewards = self.one_hot_fn(state.rewards)
            x = torch.cat([x, one_hot_rewards], dim=-1)

        binary_mask = (x.sum(dim=-1) == 0).t()

        embeddings = self.encoder(x, src_key_padding_mask=binary_mask)

        if self.encoder.training:
            embeddings = embeddings * state.rewards.unsqueeze(-1).float()
        sum_over_embeddings = embeddings.transpose(0, 1).sum(dim=1)
        return sum_over_embeddings


class LastItemStateAgg(RNNStateAgg):
    def __init__(self, embedding: CustomEmbedding, state_config, *args, **kwargs):
        super().__init__(embedding, state_config, *args, **kwargs)
        self.encoder = None
        self.state_size = embedding.embedding_dim

    def forward(self, state):
        if state.items.size(1) == 1:
            last_item = state.items[-1].view(1, -1)
        else:
            seq_lens = get_seq_len(state)
            last_idx = seq_lens - 1
            last_item = state.items.t()[torch.arange(state.items.size(1)), last_idx]

        x = self.context_embedding(last_item)
        return x.squeeze(1)


class CNNStateAgg(RNNStateAgg):
    def __init__(self, embedding: CustomEmbedding, state_config, *args, **kwargs):
        super().__init__(embedding, state_config, *args, **kwargs)

        self.encoder = torch.nn.Sequential(
            nn.Conv1d(embedding.embedding_dim, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()

        )
        self.state_size = 96

    def forward(self, state):
        max_input_len = 25
        self.check_input(state.items, state.rewards)

        x = self.context_embedding(state.items)

        # Only make this reward stuff if we use it
        if self.use_rewards_concat:
            one_hot_rewards = self.one_hot_fn(state.rewards)
            x = torch.cat([x, one_hot_rewards], dim=-1)

        transposed = x.permute(1, 2, 0)
        padded_input = F.pad(transposed, (0, max_input_len - transposed.size(-1)))
        return self.encoder(padded_input)
