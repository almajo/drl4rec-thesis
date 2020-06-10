from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions
from tqdm import tqdm

from util.in_out import read_info_file

"""

Create the data that is necessary for a simulation environment as given in https://dl.acm.org/doi/pdf/10.1145/3336191.3371801
We fill the rating-matrix via BPR-MF

Embeddings-size, default bs 256
ML: 32, bs 256
Taobao: 32, bs 256
"""
np.random.seed(123)

parser = ArgumentParser()
parser.add_argument("train")
# parser.add_argument("test")
parser.add_argument("folder_name")
args = parser.parse_args()

train_file = Path(args.train)
# test_file = Path(args.test)
folder_name = args.folder_name
output_base_dir = Path(train_file).parent

sim_dir = output_base_dir / "simulator/bpr/{}".format(folder_name)

if not sim_dir.exists():
    sim_dir.mkdir(parents=True)

####

n_dimensions = 64
batch_size = 256
num_major_iterations = 10
num_minor_iterations = 15

####

print("Creating simulation items for the dataset in {}".format(train_file))
data = pd.read_csv(train_file, na_filter=False, converters={"rating": np.float32})
# vdf = pd.read_csv(test_file, na_filter=False, converters={"rating": np.float32})

infos = read_info_file(train_file.parent)
num_items = int(infos.get("num_items")) + 1
# data = pd.concat([tdf, vdf], axis=0)
interactions = Interactions(data.userId.values, data.movieId.values,
                            timestamps=data.timestamp.values,
                            num_items=num_items)
model = ImplicitFactorizationModel(embedding_dim=n_dimensions,
                                   n_iter=num_minor_iterations,
                                   loss='bpr',
                                   use_cuda=torch.cuda.is_available(),
                                   batch_size=batch_size,
                                   learning_rate=1e-3,
                                   l2=1e-5)

test_user_ids = data.userId.unique()  # keeps order of appearance

for i in tqdm(range(num_major_iterations)):
    print("doing it number {}".format(i))
    save_dir = sim_dir / str(i)
    if not save_dir.exists():
        save_dir.mkdir()
    model.fit(interactions, verbose=True)
    torch.save(model._net.state_dict(), save_dir / "model.pkl")

    with torch.no_grad():
        scores = np.empty((len(test_user_ids), num_items), dtype=np.float32)
        for e, user in enumerate(test_user_ids):
            rating = model.predict(user)
            scores[e] = rating
        scores = torch.as_tensor(scores)
        torch.save(scores, save_dir / "raw_rating_scores.pkl")
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True)
        centered_scores = (scores - mean) / std
        torch.save(centered_scores, save_dir / "centered_scores.pkl")
