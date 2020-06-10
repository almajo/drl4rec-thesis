from pathlib import Path

import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA

base_path = Path("/home/stud/grimmalex/datasets/ml/ml-25m")
all_movies = pd.read_csv(base_path / "ratings_prep.csv", usecols=["movieId"]).squeeze()
scores = pd.read_csv(base_path / "genome_scores_prep.csv")
tags = pd.read_csv(base_path / "genome-tags.csv")
print(len(scores.movieId.unique()))
print(scores.tagId.max())
num_movies = all_movies.max()
num_tags = len(scores.tagId.unique())
print(num_movies, num_tags)
m = coo_matrix((scores.relevance.values, (scores.movieId.values, scores.tagId.values)),
               shape=(num_movies + 1, num_tags + 1)).toarray()
print(m.shape)
n_comp = 32
pca = PCA(n_components=n_comp, whiten=True, random_state=51)
transformed = pca.fit_transform(m)

print(pca.explained_variance_)
t = torch.as_tensor(transformed)
torch.save(t, base_path / f"pca_genome_embedding_{n_comp}.pkl")
