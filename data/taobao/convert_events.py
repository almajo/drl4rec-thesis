import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from data.movielens.prepare_movielens import apply_user_min_and_max_support, create_freq_sorted_index

"""
Script to create the ratings_prep file (like for ML) whose output can then be used further
User minsupport should be bigger 1, else it is none-sense to optimize some future reward
max support is limited in order to get rid of spammers. This is session-based, so it makes no sense to allow > 30 clicks

item min support should be bigger 1
"""

user_min_support = 5
user_max_support = 100
item_min_support = 5

parser = ArgumentParser()
parser.add_argument("directory")
args = parser.parse_args()

data_dir = Path(args.directory)
file = data_dir / "UserBehavior.csv"

print("Loading the dataset")
df = pd.read_csv(file, names=["userId", "movieId", "category", "rating", "timestamp"])


# MAP THE USERS TO RANGE
def reset_user_id(data):
    user_ids = np.unique(data.userId)
    user_ids.sort()
    user_id_map = {old: new for new, old in enumerate(user_ids)}

    new_user_ids = data.userId.map(lambda x: user_id_map.get(x))
    data.userId = new_user_ids.copy()
    return data


df = reset_user_id(df)
df.rating = 1.
category_info = df[["movieId", "category"]].copy()
df.drop(columns=["category"], inplace=True)

df, id_dict, categories = create_freq_sorted_index(df, category_info)
print("total-len: {}".format(len(df)))
# only use sessions which have items in the top 25k most popular items
MAX_ITEM = 25_000
df = df[df.movieId < MAX_ITEM]
print("Len after picking top 25k: {}".format(len(df)))
# Remove consecutive clicks, whatever type of event it was. this means 1,2,2,3,2,4,4 becomes 1,2,3,2,4
df.sort_values(by=["userId", "timestamp"], inplace=True)
df = df.loc[df.movieId.shift() != df.movieId]
print("len after removing consecutives: {}".format(len(df)))
print(
    "session length description before user constraints: \n{}".format(df.groupby("userId").movieId.count().describe()))
df = apply_user_min_and_max_support(df, user_min_support, user_max_support)
print("len after applying user length constraints: {}".format(len(df)))

print("ending up with {} users".format(len(df.groupby("userId").userId.max())))

sys.exit()
# subsample n users to get a more handleable space
n = 60_000
users = df.userId.unique()
print("found {} users, subsampling to {}".format(len(users), n))
indices = np.random.default_rng(0).choice(users, size=n, replace=False)
df = df.loc[df.userId.isin(indices)]

print(df.groupby("userId").movieId.count().describe())
df = reset_user_id(df)
df.to_csv(data_dir / "ratings_prep.csv", index=False)
categories.to_csv(data_dir / "categories.csv", index=False)

num_items = len(df.movieId.unique())
num_users = len(df.userId.unique())
total_num_interactions = len(df)
num_categories = len(categories.category.unique())
output = """num_items: {}
num_users: {}
num_interactions: {}
num_categories: {}
""".format(num_items, num_users, total_num_interactions, num_categories)
print(output)

with open(data_dir / "data.info", "w+") as f:
    f.write(output)
