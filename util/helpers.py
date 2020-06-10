import os
import random

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pad_sequence

from environments.recommendation_env import ReturnStateTuple


def get_seq_len(state: ReturnStateTuple):
    return state.items.t().sign().sum(dim=-1)


def flatten(l):
    return [i for x in l for i in x]


def unwrap_state(states, device):
    """
    Converts a list of multiple ReturnStateTuples of numpy arrays to one ReturnStateTuple of padded tensors
    """
    state_tuple = ReturnStateTuple(*zip(*states))
    items = [torch.from_numpy(item) for item in state_tuple.items]
    items_padded = pad_sequence(items, batch_first=False).to(device)
    if any(state_tuple.targets):
        targets = torch.as_tensor(np.concatenate(state_tuple.targets), device=device)
    else:
        targets = None

    if isinstance(state_tuple.history[0], np.ndarray):
        history = torch.from_numpy(np.stack(state_tuple.history)).to(device)
        return ReturnStateTuple(items_padded, None, history, targets)

    return ReturnStateTuple(items_padded, None, None, targets)


def set_random_seeds(random_seed):
    """Sets all possible random seeds so results can be reproduced"""
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


@torch.jit.script
def isin_2d(ar1, ar2):
    k = []
    for ind, mask in zip(ar1, ar2):
        k.append((ind[..., None] == mask).any(-1))
    return torch.stack(k)


def numpy_isin_2d(action_ids, mask_ids):
    mask_ids = np.stack([np.isin(a, m) for a, m in zip(action_ids, mask_ids)])
    return mask_ids


def user_item_list_to_coo(data, output_shape):
    user_lens = map(len, data)
    users = [[user_id] * traj_len for user_id, traj_len in enumerate(user_lens)]
    users = flatten(users)
    movies = flatten(data)
    rating = [1.] * len(movies)
    matrix = coo_matrix((rating, (users, movies)), shape=output_shape, dtype=np.float32)
    return matrix
