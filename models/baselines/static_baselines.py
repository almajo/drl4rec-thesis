import logging
from abc import ABC
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.evaluation import Evaluation
from util.helpers import flatten, user_item_list_to_coo


class StaticRecommender(Evaluation, ABC):
    name = "AbstractStatic"

    def __init__(self, train_env, k):
        Evaluation.__init__(self, k)
        self.train_env = train_env


class RandomBaseline(StaticRecommender):
    name = "Random"

    def __init__(self, train_env, k, **kwargs):
        super().__init__(train_env, k)
        self.rng = np.random.default_rng()
        self.num_items = train_env.action_space.n

    def get_eval_action(self, obs, k):
        k = self.rng.choice(np.arange(1, self.num_items), (len(obs), k), replace=False)
        return k


class PopularBaselineAdaptive(StaticRecommender):
    name = "POP-A"

    def __init__(self, train_env, k, **kwargs):
        super().__init__(train_env, k)
        data = self.train_env.base_env.data
        counter = Counter(flatten(data))
        self.top_200 = np.array([a[0] for a in counter.most_common(200)])

    def get_eval_action(self, obs, k):
        out = []
        for i in range(len(obs)):
            result = self.top_200[np.isin(self.top_200, obs[i], invert=True)][:k]
            out.append(result)
        return np.stack(out)


class PopularBaseline(StaticRecommender):
    name = "POP"

    def __init__(self, train_env, k, **kwargs):
        super().__init__(train_env, k)
        data = self.train_env.base_env.data
        counter = Counter(flatten(data))
        self.top_k = np.array([a[0] for a in counter.most_common(k)])

    def get_eval_action(self, obs, k):
        return np.stack([self.top_k for _ in range(len(obs))])


class ItemKNN(StaticRecommender):
    name = "ItemKNN"

    def __init__(self, train_env, k, tf_idf=False, log_dir=None, **kwargs):
        super().__init__(train_env, k)
        log_path = log_dir / "knn_matrix"
        if log_path is not None and log_path.exists():
            self.knn = np.load(log_path)
        else:
            self.knn = self.create_from_env(train_env, k, tf_idf)
            if log_path is not None:
                print("saving nearest neighbors to {}".format(log_dir))
                np.save(str(log_dir / "knn_matrix"), self.knn)

    def create_from_env(self, train_env, k, tf_idf=True):
        data = train_env.base_env.data

        output_shape = (len(data), train_env.action_space.n)
        item_rating_matrix = user_item_list_to_coo(data, output_shape)

        logging.info("Computing distance matrix")
        vectors = item_rating_matrix.transpose().tocsr()
        if tf_idf:
            vectors = TfidfTransformer().fit_transform(vectors)

        distance_matrix = cosine_similarity(vectors, dense_output=False)

        distance_matrix = distance_matrix.toarray()
        distance_matrix[np.isclose(distance_matrix, 1)] = 0  # shadow self-similarity for max lookup

        knn = np.argsort(distance_matrix, axis=1)[:, -(k + 20):]
        return knn

    def get_eval_action(self, obs, k):
        lookup_indices = np.array([o[-1] for o in obs])
        nearest_neighbors = self.knn[lookup_indices].flatten()
        state = np.concatenate(obs)
        not_in = np.in1d(nearest_neighbors, state, invert=True)
        nearest_neighbors = nearest_neighbors[not_in]
        return_values = np.expand_dims(nearest_neighbors[:k], 0)
        return return_values
