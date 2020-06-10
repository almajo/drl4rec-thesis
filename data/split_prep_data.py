from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

"""
Takes the base_path to the dataset as input, e.g. ml/ml-1m and splits the ratings_prep.csv file into train/valid/test
"""

parser = ArgumentParser()
parser.add_argument("dir")
args = parser.parse_args()

data_dir = Path(args.dir)
test_split_ratio = 0.2
valid_split_ratio = 0.3  # 10 percent of train split
file_path = data_dir / "ratings_prep.csv"

data = pd.read_csv(file_path, na_filter=False, dtype={"userId": int, "movieId": int, "rating": float})

max_user_id = data.userId.max()
num_users = max_user_id + 1
split_id = num_users - num_users * test_split_ratio

# Create a first split
train_full = data.loc[data.userId < split_id]
test = data.loc[data.userId >= split_id]

train_full.to_csv(data_dir / "train_full.csv", index=False)
test.to_csv(data_dir / "test.csv", index=False)

# split the train data into train and valid, where valid is for model-choice evaluation
valid_split_id = train_full.userId.max() * (1 - valid_split_ratio)
valid = train_full.loc[train_full.userId >= valid_split_id]
train_split = train_full.drop(valid.index)
valid.to_csv(data_dir / "valid_split.csv", index=False)
train_split.to_csv(data_dir / "train_split.csv", index=False)
