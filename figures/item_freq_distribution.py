# import matplotlib as mpl
import numpy as np

from data.movielens.prepare_movielens import create_freq_sorted_index
from figures.visualize_results import *


# mpl.use("TkAgg")


def create_session_len_plot(path):
    fig = plt.figure(figsize=[6.4, 4.8])
    # order = ["ml/ml-1m"]
    order = ["ml/ml-1m", "ml/ml-25m", "taobao"]
    ax = None
    for d in order:
        if "ml" in d:
            data = path / d / "ratings.csv"
            df = pd.read_csv(data, usecols=[0, 1])
        else:
            if (path / d / "session_len.csv").exists():
                df = pd.read_csv(path / d / "session_len.csv")
            else:
                data = path / d / "UserBehavior.csv"
                df = pd.read_csv(data, header=None, names=["userId", "movieId", "category", "rating", "timestamp"])
                print(len(df))
                df, _ = create_freq_sorted_index(df)
                MAX_ITEM = 25_000
                df = df[df.movieId < MAX_ITEM]
                print("Len after picking top 25k: {}".format(len(df)))
                # Remove consecutive clicks, whatever type of event it was. this means 1,2,2,3,2,4,4 becomes 1,2,3,2,4
                df.sort_values(by=["userId", "timestamp"], inplace=True)
                df = df.loc[df.movieId.shift() != df.movieId]

                df = df[["userId", "movieId"]]
                print("len after removing consecutives: {}".format(len(df)))
                df.to_csv(path / d / "session_len.csv")

        session_lens = df.groupby("userId").movieId.count()
        session_lens = session_lens[session_lens < 500]
        print("loading {}".format(d))
        label = (path / d).name
        ax = sns.distplot(session_lens, hist=False,
                          label=label.upper() if "ml" in label else label.title())
    # plt.yscale("log")
    plt.axvline(300, color="black", linestyle="--")
    plt.axvline(100, color="black", linestyle="--")
    plt.text(110, 0.03, '$t_{max}^{Taobao}$')
    plt.text(310, 0.03, '$t_{max}^{ML}$')

    # plt.axhline(y=5, color="k", linestyle='--')
    # plt.text(1, 5.5, '$i_{min} = 5$')
    plt.legend(loc="best")
    plt.xlabel("Trajectory Length")
    plt.ylabel("Probability Density")
    plt.tight_layout()
    # plt.show()
    save_plot(path, "session_len_dist")


def create_item_freq_plot(path):
    fig = plt.figure(figsize=[6.4, 4.8])
    order = ["ml/ml-1m", "ml/ml-25m", "taobao"]
    for d in order:
        d = path / d / "item_counts.csv"
        print("loading {}".format(d))
        df = pd.read_csv(d, usecols=["counts"], squeeze=True)
        label = d.parent.name
        # plt.hist(df.values, bins=100, log=True)
        # bins = len(df) // 20
        bins = np.arange(0, len(df), 100)
        counts, bin_edges = np.histogram(df.values, bins)
        print(df.describe())
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
        # ax = sns.lineplot(bin_centres, counts, label=label)
        # ax = sns.distplot(df.values, hist=True, kde=False, hist_kws={"histtype": "step"}, bins=bins,
        #                   label=label.upper() if "ml" in label else label.title())
        ax = plt.plot(bin_centres, counts, label=label.upper() if "ml" in label else label.title())
    plt.xscale("log")
    plt.yscale("symlog")
    # plt.axhline(y=5, color="k", linestyle='--')
    # plt.text(1, 5.5, '$i_{min} = 5$')
    plt.legend(loc="best")
    plt.xlabel("Number of Items")
    plt.ylabel("Frequency")
    plt.tight_layout()
    save_plot(path, "item_freq_dist_new")


if __name__ == '__main__':
    data_path = Path("/home/stud/grimmalex/datasets")
    create_item_freq_plot(data_path)
    # create_session_len_plot(data_path)
