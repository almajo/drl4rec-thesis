from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def create_freq_sorted_index(ratings, *others):
    movie_ids_sorted = list(ratings.groupby("movieId").userId.count().sort_values(ascending=False).index)
    movie_ids_new_index_dict = {old: new for new, old in enumerate(movie_ids_sorted, 1)}
    # Set new_id in movieId
    ratings.loc[:, "movieId"] = ratings.movieId.map(lambda x: movie_ids_new_index_dict[x])

    # Update Ids of contents, remove items which are not in the dataset
    old = set(movie_ids_new_index_dict.keys())
    new_dfs = []
    for content in others:
        ids_of_items_in_logs = content[content["movieId"].isin(old)].copy()
        ids_of_items_in_logs.loc[:, "movieId"] = ids_of_items_in_logs.movieId.map(
            lambda x: movie_ids_new_index_dict[x])
        assert not ids_of_items_in_logs.movieId.isna().any()
        new_dfs.append(ids_of_items_in_logs)
    return [ratings, movie_ids_new_index_dict, *new_dfs]


def apply_item_min_support(df, min_sup):
    movie_counts = df.groupby("movieId").timestamp.count()
    applied_min_sup = movie_counts[movie_counts >= min_sup]
    return df[df.movieId.isin(applied_min_sup.index)]


def apply_user_min_and_max_support(df, min_sup, max_len):
    user_movie_counts = df.groupby("userId").movieId.count()
    applied_min_sup = user_movie_counts[(min_sup <= user_movie_counts) & (user_movie_counts <= max_len)]
    filtered_users = df[df.userId.isin(applied_min_sup.index)]
    return filtered_users


def filter_ml25m_to_sessions_with_genome_ids_only(df, scores):
    movies_with_genome = set(scores.movieId.unique())
    trajectories = df.groupby("userId").movieId.agg(list)
    c = trajectories.map(lambda x: all([s in movies_with_genome for s in x]))
    user_ids = c[c].index
    all_genome_sessions = df[df.userId.isin(user_ids)]
    return all_genome_sessions


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    user_min_support = 20
    user_max_support = 300
    item_min_support = 5

    p = Path(args.directory)
    rating_df = pd.read_csv(p / "ratings.csv", dtype={"movieId": np.int32, "userId": np.int32, "rating": np.float32,
                                                      "timestamp": np.int64})
    content_df = pd.read_csv(p / "movies.csv", dtype={"movieId": np.int32})

    rating_df = apply_item_min_support(rating_df, min_sup=item_min_support)
    rating_df = apply_user_min_and_max_support(rating_df, user_min_support, user_max_support)
    if "25" not in str(p):
        rating_df, index_map, content_df = create_freq_sorted_index(rating_df, content_df)
    else:
        genome_scores = pd.read_csv(p / "genome-scores.csv", dtype={"movieId": np.int32, "tagId": np.int32})
        rating_df = filter_ml25m_to_sessions_with_genome_ids_only(rating_df, genome_scores)
        rating_df, index_map, content_df, genome_df = create_freq_sorted_index(rating_df, content_df, genome_scores)
        genome_df.to_csv(Path(args.directory) / "genome_scores_prep.csv", index=False)

    # Sort data by timestamp
    rating_df.sort_values(by=["userId", "timestamp"], inplace=True)

    # Store index map in links file
    index = pd.Series(index_map)
    index = index.rename("frequency")
    if "25" in str(p):
        links = pd.read_csv(p / "links.csv")
        links = links.join(index, on="movieId")
        links.to_csv(Path(args.directory) / "links_prep.csv", index=False)

    assert rating_df.notna().all().all()
    rating_df.rating = 1
    rating_df.astype(np.long)

    rating_df.to_csv(Path(args.directory) / "ratings_prep.csv", index=False)
    content_df.to_csv(Path(args.directory) / "content_prep.csv", index=False)

    num_items = len(rating_df.movieId.unique())
    num_users = len(rating_df.userId.unique())
    total_num_interactions = len(rating_df)

    output = f"""num_items: {num_items}
num_users: {num_users}
num_interactions: {total_num_interactions}
density: {total_num_interactions / (num_items * num_users)}"""
    print(output)

    with open(Path(args.directory) / "data.info", "w+") as f:
        f.write(output)
