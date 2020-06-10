import torch
from torch.distributions import Categorical


def sample_pytorch_dist(logits, sample_size=1):
    d = Categorical(logits=logits)
    next_policy_indices = d.sample((sample_size,))
    log_probs = d.log_prob(next_policy_indices)
    return next_policy_indices, log_probs, d


def sample_without_replacement(probs, k):
    """
    Samples without replacement via reservoir sampling (exponential/Gumbel trick). Faster than pytorch routines
    :param probs: probability-vector
    :param k: number of samples to draw per last-dim
    :return: Samples of shape like input, but last dimension size k
    """
    w = -torch.log(probs).uniform_() / (probs + 1e-10)
    return torch.topk(w, k, largest=False)[1]


def sample_without_replacement_from_log_probs(log_probs, k):
    samples = sample_without_replacement(log_probs.exp(), k)
    log_probs = log_probs.gather(-1, samples)
    return samples, log_probs
