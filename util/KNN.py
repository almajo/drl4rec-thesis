import logging

# import nmslib
import torch

logger = logging.getLogger(__name__)

knn_variants = {
    "exact": 15000,
    "slow": 1000,
    "medium": 80,
    "fast": 20
}


def state_action_concat(state, actions):
    """
    :param state: tensor (batch_size, state_dim)
    :param actions: tensor (batch_size, num_actions, action_dim)
    :return: concatenated version tensor (num_actions * batch_size, state_dim + action_dim)
    """
    assert state.ndimension() == 2 and actions.ndimension() == 3
    state = state.unsqueeze(1).expand(-1, actions.size(1), -1)
    state_actions = torch.cat([state, actions], dim=-1)
    return state_actions


@torch.jit.script
def masking_select(indices, mask):
    if indices.size(0) == 1:
        ind = indices[0]
        mask = (ind[..., None] != mask[0]).all(-1)
        if not mask.all():
            new = ind[mask]
            ind[:len(new)] = new
            ind[-(len(ind) - len(new)):] = 0
        return indices
    new_indices = torch.zeros_like(indices)
    for e, (ind, m) in enumerate(zip(indices, mask)):
        new_inds = ind[(ind[..., None] != m).all(-1)]
        new_indices[e][:len(new_inds)] = new_inds
    return new_indices


#
# class FastApproximateLookup:
#     def __init__(self, vectors: np.ndarray, embedding_lookup):
#         index = nmslib.init(space="l2")
#         index.addDataPointBatch(vectors)
#         index_time_params = {'M': 30, 'efConstruction': 100, 'post': 0}
#         query_time_params = {'efSearch': 100}
#         index.createIndex(index_time_params, print_progress=True)
#         index.setQueryTimeParams(query_time_params)
#         self.index = index
#
#         t = time()
#         ids = self.index.knnQueryBatch(vectors, k=1)
#         logger.info("Time: {}".format(time() - t))
#         idx = torch.as_tensor([i[0][0] for i in ids], dtype=torch.float)
#         a = torch.arange(len(idx), dtype=torch.float)
#         prec = idx == a
#         logger.info("Recall: {}".format(prec.sum().float() / prec.size(0)))
#
#         self.vector_lookup = embedding_lookup
#
#     def lookup(self, proto_vectors, k, embedding_matrix=None, mask: torch.Tensor = None):
#         if not isinstance(proto_vectors, np.ndarray):
#             proto_vectors = proto_vectors.detach().cpu().numpy()
#         ids = self.index.knnQueryBatch(proto_vectors, k=k)
#         ids = [i[0] for i in ids]
#         idx = torch.as_tensor(ids, dtype=torch.long, device=embedding_matrix.device)
#         if mask is not None:
#             idx = masking_select(idx, mask)
#         vectors = self.vector_lookup(idx)
#
#         return idx, vectors


class MatmulLookup:
    def __init__(self, matrix):
        self.neighbors = matrix

    def lookup(self, proto_vectors, k, mask=None):
        distances = torch.cdist(proto_vectors, self.neighbors)
        # mask distances directly
        if mask is not None:
            distances.scatter_(1, mask.to(distances.device), float('inf'))
        # Zero is not a valid neighbor
        distances[..., 0] = float("inf")
        top_k = torch.topk(distances, k, largest=False, sorted=True).indices.squeeze(-1)
        return top_k


def matmul_lookup(proto_vectors, weight_matrix, k):
    distances = torch.cdist(proto_vectors, weight_matrix)
    distances[..., 0] = float("inf")
    top_k = torch.topk(distances, k, largest=False).indices
    return top_k
