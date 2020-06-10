import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from util.helpers import user_item_list_to_coo
from util.in_out import load_obj, save_obj


class Tree:
    def __init__(self, tree_depth=None, rating_matrix=None, data_set="ml-25m", path=None):
        """
        Rating matrix is the user x item raw rating matrix
        """
        self.path = path
        if rating_matrix is not None or not path.exists():
            logging.info("Creating Clustering tree from scratch")
            self.item_num = rating_matrix.shape[-1]
            self.child_num = math.ceil(self.item_num ** (1 / tree_depth))
            self.bc_dim = tree_depth
            self.rating_matrix = rating_matrix.T.tocsr()
            self.data_set = data_set
            self.construct_tree()
            del self.rating_matrix
        else:
            logging.info("Loading clustering tree from disk")
            obj = load_obj(self.path)
            self.item_num = len(obj['trajectory_to_id'])
            self.child_num = obj['child_num']
            self.bc_dim = obj['bc_dim']
            self.rating_matrix = None
            self.data_set = obj['data_set']
            self.id_to_trajectory = obj['id_to_trajectory']
            self.trajectory_to_id = obj['trajectory_to_id']

        self.no_item_leaves = np.ceil(self.item_num ** (1 / self.bc_dim)) ** self.bc_dim - self.item_num
        logging.info("Number of NO-ITEM leaves: {}".format(self.no_item_leaves))

    def construct_tree(self):
        self.id_to_trajectory, self.trajectory_to_id = self.build_mapping()
        obj = {'id_to_trajectory': self.id_to_trajectory, 'trajectory_to_id': self.trajectory_to_id,
               'child_num': int(self.child_num), 'bc_dim': self.bc_dim, 'data_set': self.data_set}
        save_obj(obj, str(self.path))
        return self

    def build_mapping(self):
        id_to_trajectory = defaultdict(list)
        id_to_vector = self.rating_matrix
        self.hierarchical_code(list(range(id_to_vector.shape[0])),
                               (0, int(self.child_num ** self.bc_dim)),
                               id_to_trajectory, id_to_vector)
        # reverse dict
        trajectory_to_id = {tuple(v): k for k, v in id_to_trajectory.items()}
        return id_to_trajectory, trajectory_to_id

    def pca_clustering(self, item_list, id_to_vector):
        if len(item_list) <= self.child_num:
            return [[item] for item in item_list] + [[] for _ in range(self.child_num - len(item_list))]

        data = id_to_vector[item_list]
        pca = TruncatedSVD(n_components=1, random_state=42)
        pca.fit(data)
        w = pca.components_[0]
        item_to_projection = [(item, np.dot(id_to_vector[item].toarray(), w)) for item in item_list]
        result = sorted(item_to_projection, key=lambda x: x[1])

        item_list_assign = []
        list_len = int(math.ceil(len(result) * 1.0 / self.child_num))
        non_decrease_num = self.child_num - (self.child_num * list_len - len(result))
        start = 0
        end = list_len
        for i in range(self.child_num):
            item_list_assign.append([result[j][0] for j in range(start, end)])
            start = end
            if i >= non_decrease_num - 1:
                end = end + list_len - 1
            else:
                end = end + list_len
        return item_list_assign

    def hierarchical_code(self, item_list, code_range, id_to_code, id_to_vector):
        if len(item_list) < 2:
            return

        item_list_assign = self.pca_clustering(item_list, id_to_vector)
        for i, cluster in enumerate(item_list_assign):
            for item_idx in cluster:
                id_to_code[item_idx].append(i)
        range_len = int((code_range[1] - code_range[0]) / self.child_num)
        for i in range(self.child_num):
            self.hierarchical_code(item_list_assign[i],
                                   (code_range[0] + i * range_len, code_range[0] + (i + 1) * range_len), id_to_code,
                                   id_to_vector)

    def get_action_to_trajectory(self, trajectory):
        if isinstance(trajectory, list) or isinstance(trajectory, np.ndarray):
            trajectory = tuple(trajectory)
        idx = self.trajectory_to_id[trajectory]
        return idx

    def get_trajectory_for_id(self, idx):
        traj = self.id_to_trajectory.get(idx)
        return traj


def create_clustering_tree(environment, tree_depth, overwrite_tree=False):
    # Create sparse rating matrix
    data_name = environment.id
    base_path = Path(os.path.realpath(__file__)).parent / "resources"
    path = base_path / "pca_tree-{}.pkl".format(data_name)
    print(path)
    if not path.exists() or overwrite_tree:
        data = environment.base_env.data,
        if isinstance(data, tuple):
            data = data[0]
        item_num = environment.action_space.n,
        if isinstance(item_num, tuple):
            item_num = item_num[0]
        shape = (len(data), item_num)
        matrix = user_item_list_to_coo(data, shape)
        rating_matrix = matrix
        # Create tree
        tree = Tree(tree_depth, rating_matrix, data_name, path)
    else:
        tree = Tree(path=path)

    return tree
