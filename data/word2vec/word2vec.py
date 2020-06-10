import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.manifold import TSNE

from util.in_out import read_info_file

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorboard as tb
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class WordVectorDumpCallback(CallbackAny2Vec):
    def __init__(self, data_dir: Path, emb_dim):
        self.dir = data_dir / "word2vec" / "{}".format(emb_dim)
        self.dir.mkdir(exist_ok=True, parents=True)
        self.epoch = 0
        self.last_loss = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        loss_diff = loss - self.last_loss
        self.last_loss = loss
        if self.epoch % 50 == 0:
            model.wv.save(str(self.dir / "{}-{}.wv".format(self.epoch, loss_diff)))


class Word2VecCallback(CallbackAny2Vec):
    '''
    Callback to print loss after each epoch.
    '''

    def __init__(self, tb_writer=None):
        self.writer = tb_writer
        self.epoch = 0
        self.last_loss = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        if self.epoch % 20 == 0:
            if self.writer is not None:
                self.writer.add_scalar("word2vec/loss", loss - self.last_loss,
                                       global_step=self.epoch)
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.last_loss))

        self.last_loss = loss
        self.epoch += 1


def visualize(model):
    X = model.wv[model.wv.vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()


def train_word2vec(base_dir, train_file, embedding_dim, num_iters):
    train_data_raw = pd.read_csv(train_file, usecols=["movieId", "userId"], na_filter=False)
    try:
        max_id = read_info_file(train_file.parent).get("num_items") + 1
    except FileNotFoundError:
        max_id = read_info_file(train_file.parent.parent).get("num_items") + 1

    max_id = int(max_id)

    train_data = train_data_raw.groupby("userId").agg(list).squeeze()

    # Learn an unknown token embedding. Therefore we sample 50 items randomly from the long tail and mask them to one vector
    items_train = np.unique(train_data_raw.movieId.values.ravel())
    candidates = items_train[:int(len(items_train) / 2)]
    unk_items = np.random.choice(candidates, 50, replace=False)
    print("Replacing {} items with the unknown token symbol".format(unk_items))
    unk_id = max_id + 1
    fn = np.vectorize(lambda x: str(x) if x not in unk_items else str(unk_id))
    train_data = train_data.map(lambda x: fn(x).tolist())
    string_corpus = train_data.values.tolist()
    print(len(list(filter(lambda x: str(unk_id) in x, string_corpus))))
    logging.info("Pretraining the word2vec model")

    word2vec_dir = base_dir / "word2vec"
    word2vec_dir.mkdir(exist_ok=True)
    for dim in embedding_dim:
        dim_dir = word2vec_dir / str(dim)
        dim_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(dim_dir)
        w2v = Word2Vec(string_corpus,
                       size=dim,
                       compute_loss=True,
                       callbacks=[Word2VecCallback(writer),
                                  WordVectorDumpCallback(dim_dir, embedding_dim)],
                       min_count=1,
                       workers=32,
                       iter=num_iters,
                       window=3
                       )
        wv = w2v.wv
        embedding_tensor = torch.zeros(max_id, dim, dtype=torch.float32)
        for str_id in map(str, items_train):
            if str_id in wv:
                embedding_tensor[int(str_id)] = torch.as_tensor(wv[str_id])
        empty_indices = set(range(1, max_id)) - set(items_train)
        embedding_tensor[torch.as_tensor(list(empty_indices), dtype=torch.long)] = torch.as_tensor(wv[str(unk_id)])
        torch.save(embedding_tensor, word2vec_dir / "tensor_{}.pkl".format(dim))
        writer.add_embedding(embedding_tensor, tag=str(dim))


if __name__ == '__main__':
    base = Path("/home/stud/grimmalex/datasets/taobao_new")
    train_word2vec(base, base / "train_split.csv",
                   embedding_dim=[16, 32, 64],
                   num_iters=10_000)
