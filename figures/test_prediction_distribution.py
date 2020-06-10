import pickle
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import ScalarFormatter

mpl.use("PDF")

import matplotlib.pyplot as plt
import numpy as np
from figures.visualize_results import save_plot, nice_fonts

plt.style.use("seaborn-paper")
mpl.rcParams.update(nice_fonts)

num_item_dict = {
    "taobao": 24999,
    "ml-1m": 3403,
    "ml-25m": 13461
}

DATAPATH = Path("/home/stud/grimmalex/datasets")


def get_num_items(path):
    p = str(path)
    if "taobao" in p:
        n = num_item_dict["taobao"]
        name = "taobao"
    elif "ml-1m" in p:
        n = num_item_dict["ml-1m"]
        name = "ml/ml-1m"
    elif "ml-25m" in p:
        n = num_item_dict["ml-25m"]
        name = "ml/ml-25m"

    else:
        raise KeyError("Could not find the dataset in the path string")
    return n + 1, name


def load_dataset_for_real_distribution(data_name):
    p = DATAPATH / data_name
    df = pd.read_csv(p / "test.csv", usecols=["movieId"], squeeze=True)

    ids = df.values.flatten()
    c = Counter(ids)
    ids, counts = list(zip(*c.items()))
    item_id, values = np.array(ids), np.array(counts)
    ind_sort = np.argsort(item_id)
    item_id = item_id[ind_sort]
    values = values[ind_sort]

    values = values / values.sum()
    return item_id, values


def plot_predictions(log_folders: list, bin_size=1000, log_scale=False):
    num_items, data_name = get_num_items(log_folders[0])
    bins = np.arange(0, num_items, bin_size)

    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    #
    # real_distribution_id, real_counts = load_dataset_for_real_distribution(data_name)
    # plt.hist(real_distribution_id, weights=real_counts, label="Real", linewidth=1.2)
    i = 0
    for path in log_folders:
        path = path / "test_prediction_counter.pkl"
        with open(str(path), "rb") as f:
            loader = pickle.load(f)
        item_id, values = list(zip(*loader.items()))
        item_id, values = np.array(item_id), np.array(values)
        ind_sort = np.argsort(item_id)
        item_id = item_id[ind_sort]
        values = values[ind_sort]
        method = path.parent
        try:
            int(method.name)
            method = method.parent.name
        except ValueError:
            method = method.name

        if method.islower():
            method = method.upper()
        histtype = "bar" if i == 0 else "step"
        values = values / values.sum()
        # print(item_id.shape, values.shape)
        # new_ar = np.array(flatten([[idx] * w for idx, w in zip(item_id, values)]))
        # sns.distplot(new_ar)
        plt.hist(item_id, weights=values, bins=bins, label=method, histtype=histtype, linewidth=1.2)
        i += 1

    if log_scale:
        plt.yscale('log', nonposy="clip")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.xlabel("Item ID")
    plt.ylabel("Percentage of Total Predictions")

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="best")
    plt.minorticks_off()
    save_plot(log_folders[0], "prediction_distribution_comp")
    print("Saved plot to {}".format(log_folders[0]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) for p in args.folders]
    plot_predictions(paths, log_scale=args.log)
