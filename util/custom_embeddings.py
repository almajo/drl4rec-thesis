import torch
import torch.nn.functional as F
from sklearn.preprocessing import minmax_scale
from torch import nn


def get_embedding(config, **kwargs):
    emb_type = config["type"]
    if emb_type == "standard":
        return StandardEmbedding(config, **kwargs)
    elif emb_type == "word2vec_fixed":
        weight = torch.load(config["w2v_context_path"])
        scaled_weight = minmax_scale(weight, feature_range=(-1, 1), axis=1)
        embedding = torch.nn.Embedding.from_pretrained(torch.as_tensor(scaled_weight, dtype=torch.float32),
                                                       padding_idx=0, freeze=True)
        return embedding
    elif emb_type == "genome":
        weight = torch.load(config["tag_genome_embedding_path"])
        # scaled_weight = minmax_scale(weight, feature_range=(-1, 1), axis=1)
        embedding = torch.nn.Embedding.from_pretrained(torch.as_tensor(weight, dtype=torch.float32),
                                                       padding_idx=0, freeze=True)
        return embedding
    elif emb_type == "word2vec_cont":
        weight = torch.load(config["w2v_context_path"])
        embedding = torch.nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=False,
                                                       sparse=config["sparse_grad"])
        return embedding
    elif emb_type == "content":
        return ContentEmbedding(config, **kwargs)
    elif emb_type == "one_hot":
        return OneHotEmbedding(config, **kwargs)
    else:
        raise ModuleNotFoundError("Could not find the embedding you specified in Embedding.type")


class CustomEmbedding(nn.Embedding):
    def __init__(self, embedding_config, del_weight=False, tanh_wrap=False, **kwargs):
        super().__init__(embedding_config["num_items"], embedding_config["embedding_dim"],
                         sparse=embedding_config["sparse_grad"],
                         scale_grad_by_freq=embedding_config["scale_by_freq"],
                         **kwargs)
        self.config = embedding_config
        self.weight.requires_grad_(not embedding_config["freeze"])
        self.wrap_fn = torch.tanh if tanh_wrap else lambda x: x
        if del_weight:
            del self.weight

    @property
    def device(self):
        return self.weight.device


class StandardEmbedding(CustomEmbedding):

    def __init__(self, config,
                 **kwargs):
        if "padding_idx" in kwargs:
            del kwargs["padding_idx"]
        super().__init__(config, padding_idx=0, **kwargs)

    def set_weight(self, weight, continue_training=False):
        self.weight = torch.nn.Parameter(weight)
        self.weight.requires_grad_(continue_training)
        self.num_embeddings = weight.size(0)
        self.embedding_dim = weight.size(-1)


class OneHotEmbedding(CustomEmbedding):
    def __init__(self, config, num_actions, **kwargs):
        if "padding_idx" in kwargs:
            del kwargs["padding_idx"]
        super().__init__(config, padding_idx=None, **kwargs)
        weight = F.one_hot(torch.arange(num_actions))
        self.set_weight(weight.float(), continue_training=False)

    def set_weight(self, weight, continue_training=False):
        self.weight = torch.nn.Parameter(weight)
        self.weight.requires_grad_(continue_training)
        self.num_embeddings = weight.size(0)
        self.embedding_dim = weight.size(-1)


class ContentEmbedding(StandardEmbedding):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        load_path = config["tag_genome_embedding_path"]
        embedding_tensor = torch.load(load_path)
        last_dim_unk_tensor = torch.zeros(1, embedding_tensor.size(-1))
        embedding_tensor = torch.cat([embedding_tensor, last_dim_unk_tensor], dim=0)
        self.set_weight(embedding_tensor, continue_training=False)
