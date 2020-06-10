from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl

mpl.use("PDF")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-paper")
nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 16,
    "font.size": 16,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 300
}

mpl.rcParams.update(nice_fonts)


def load_metrics_to_df(path, file_name="valid_metrics", max_episode_limit=10_000):
    metrics = []
    for metrics_file in path.glob("**/{}.csv".format(file_name)):
        # print(metrics_file)
        df = pd.read_csv(metrics_file, header=None, dtype={0: int})
        names = ["Epoch", "Return", "HitRate", "List-Diversity", "Item-Diversity", "Mean-Duration"]
        df.columns = names
        df["Experiment"] = [metrics_file.parent.parent.parent.parent.name] * len(df)
        df["Method"] = [str.upper(metrics_file.parent.parent.name)] * len(df)
        df["Seed"] = [metrics_file.parent.name] * len(df)
        df["Dataset"] = [str.upper(metrics_file.parent.parent.parent.name)] * len(df)

        if "test" in file_name:
            df = df.tail(1)
        metrics.append(df)
    total = pd.concat(metrics)
    # trim to a max epoch
    total = total.loc[total.Epoch <= max_episode_limit]

    total.reset_index(drop=True, inplace=True)
    print(total)
    total = total.sort_values(by=["Experiment", "Method", "Seed"])
    total.to_csv(path / "total_{}.csv".format(file_name), index=False)

    save_mean_best_performance(total, path / "mean_{}_table.csv".format(file_name))

    return total


def save_mean_best_performance(df, save_path):
    # save the mean/std value to a table to have one-glance comparison to baselines
    group = ["Experiment", "Method", "Dataset", "Seed"]
    max_total = df.loc[df.groupby(group)["Return"].idxmax()]
    max_total.drop(columns=["Epoch"], inplace=True)
    means_perf = max_total.groupby(["Experiment", "Method", "Dataset"]).agg(["mean", "std"])
    means_perf.sort_values(by=["Dataset", "Experiment"], inplace=True)
    means_perf.reset_index(inplace=True)

    means_perf.fillna(0, inplace=True)
    formatter_fn = lambda x: "{:.3f} ({:.3f})".format(x["mean"], x["std"])
    columns = list(means_perf.columns.levels[0][:-3])
    aggs = []
    for col in columns:
        aggs.append(means_perf.apply(lambda x: formatter_fn(x[col]), axis=1))
    aggs = pd.concat(aggs, axis=1)

    total_table = pd.concat([means_perf.loc[:, ["Experiment", "Method", "Dataset"]], aggs], axis=1)
    total_table.columns = ["Experiment", "Method", "Dataset"] + columns

    total_table.to_csv(save_path, index=False)


def create_metrics_plot(df, experiment, dataset):
    """
    This asserts the base_dir provided was the data_set_directory, i.e. .../ml-1m
    """
    df = df[(df["Experiment"] == experiment) & (df["Dataset"] == dataset)]
    if len(df) == 0:
        return
    title = "{} - {}".format(dataset, experiment)

    fig, (ax1, ax2) = plt.subplots(2, constrained_layout=True)
    # fig.subplots_adjust(top=0.8)
    sns.lineplot(x="Epoch", seed=12, y="Return", hue="Method", data=df, ax=ax1, legend=False)
    sns.lineplot(x="Epoch", seed=12, y="List-Diversity", hue="Method", data=df, ax=ax2)

    h, l = ax2.get_legend_handles_labels()
    ax2.legend_.remove()
    legend = plt.legend(h, l, bbox_to_anchor=(1, 1), loc="upper left", frameon=False)
    ax1.label_outer()
    # plt.suptitle(title)

    return "metrics_{}_{}".format(experiment, dataset), legend


def direct_factor_comparison(df, factor, y="Return"):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey="all")
    methods = df.Method.unique()
    for i, (method, ax) in enumerate(zip(methods, [ax1, ax2, ax3, ax4, ax5])):
        data = df.loc[df.Method == method]
        sns.lineplot(x="Epoch", y=y, seed=16, hue=factor, data=data, ax=ax)
        ax.set_title(method)
        if i < 4:
            ax.legend_.remove()
    # Hide inner x labels
    ax1.xaxis.label.set_visible(False)
    ax2.xaxis.label.set_visible(False)

    # create legend in empty subfigure space
    h, l = ax5.get_legend_handles_labels()
    ax5.legend_.remove()
    legend = ax6.legend(h, l, borderaxespad=0, frameon=False, loc="center")
    ax6.axis("off")

    # fig.tight_layout()
    # plt.show()
    return "experiment_{}_{}_comp".format(df[factor].values[0], y.lower()), legend


def save_plot(path, file_name, **kwargs):
    plt.savefig(path / "{}.png".format(file_name), bbox_inches='tight', **kwargs)
    plt.savefig(path / "{}.pdf".format(file_name), bbox_inches='tight', format="pdf", dpi=300, **kwargs)


def load_train_df(path, max_episode_num=10_000):
    file_name = "training_tb_metrics.csv"
    metrics_file = path / file_name
    if not metrics_file.exists():
        raise FileNotFoundError("Could not find the metrics file. Generate it first")

    print("loading training csv file")
    df = pd.read_csv(metrics_file)
    # Trim to max episode
    df = df.loc[df.Epoch <= max_episode_num]

    # Only take the mean of many logs per episode
    df = df.groupby(["Epoch", "Method", "Experiment", "Seed", "Dataset"]).mean().reset_index()

    df.Method = df.Method.map(str.upper)
    df.Dataset = df.Dataset.map(str.upper)
    return df


def metrics_progress_plot(df, path, sliding_window=5, y="Return", df_type="train"):
    fig = plt.figure(figsize=[6.4, 4.8])
    df = df[df.Epoch >= 20]
    rolling_metric = df.groupby(["Method", "Seed", "Dataset"])[y].apply(
        lambda x: x.ewm(span=sliding_window).mean())
    df.loc[:, y] = rolling_metric

    df.Epoch = df.Epoch // 89
    print(df.iloc[0])
    ax = sns.lineplot(x="Epoch", y=y, seed=16, hue="Method", data=df)
    plt.xlabel("Iteration")
    handles, labels = ax.get_legend_handles_labels()
    try:
        index_of = labels.index("Method")
        handles.pop(index_of)
        labels.pop(index_of)
    except ValueError:
        pass
    ax.legend_.remove()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)  # , bbox_to_anchor=(1,0.7))
    plt.minorticks_off()
    plt.tight_layout()
    save_plot(path, "{}_{}".format(df_type, y.lower()))


def train_return_q_value_plot(df, path, sliding_window=200):
    df = df[(df.Seed == 0) & (df.Epoch >= 20)]
    rolling_return = df.groupby(["Method", "Experiment", "Seed", "Dataset"])["Return"].apply(
        lambda x: x.ewm(span=sliding_window).mean())
    df.loc[:, "Return"] = rolling_return

    rolling_q = df.groupby(["Method", "Experiment", "Seed", "Dataset"])["Q-values"].apply(
        lambda x: x.ewm(span=sliding_window).mean())
    df.loc[:, "Q-values"] = rolling_q
    df = df[~df["Q-values"].isna()]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))
    sns.lineplot(x="Epoch", y="Return", seed=16, hue="Method", data=df, ax=ax1, estimator=None, legend=False)
    sns.lineplot(x="Epoch", y="Q-values", seed=16, hue="Method", data=df, ax=ax2, estimator=None)

    handles, labels = ax2.get_legend_handles_labels()
    try:
        index_of = labels.index("Method")
        handles.pop(index_of)
        labels.pop(index_of)
    except ValueError:
        pass
    ax2.legend_.remove()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax2.legend(handles, labels, loc='lower right')  # , bbox_to_anchor=(1,0.7))
    if df["Q-values"].max() > 100:
        ax2.set_yscale("log")
    plt.tight_layout()
    plt.minorticks_off()
    plt.xlabel("Episode")
    save_plot(path, "train_return_and_q_exp")


def create_train_all_exp_plot(df, path):
    print("Creating training plot")
    g = sns.FacetGrid(df, row="Experiment", col="Dataset", hue="Method", margin_titles=True)
    g = g.map(sns.lineplot, "Epoch", "Return")
    # This only shows the values as a title in row and col titles instead of key=value
    for ax in g.axes.flat:
        plt.setp(ax.texts, text="")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.add_legend()

    save_plot(path, "train_return")


def online_eval_bar_plot(path_list, reward_type="item"):
    """
    Create the figure in the online evaluation
    """

    df = []
    for pa in path_list:
        print(pa)
        df.append(pd.read_csv(Path(pa) / "{}_bpr_test_new.csv".format(reward_type)))
    df = pd.concat(df)

    df.loc[:, "Dataset"] = df.Dataset.map(lambda x: str(x).upper() if "ml" in x else str(x).title())
    df.loc[:, "Method"] = df.Method.map(lambda x: str(x).upper() if x in ["Dueling", "Random", "Correction"] else x)
    df = df.loc[df.Method != "LIRD"]
    df.loc["POP-A" == df.Method, "Method"] = df.loc["POP-A" == df.Method, "Method"].map(lambda x: "POP")

    df.sort_values(["Dataset", "Return"], inplace=True)
    result = df.groupby(["Method"])['Return'].mean().reset_index().sort_values('Return')
    ax = sns.barplot(x="Method", y="Return", hue="Dataset", data=df, capsize=0.2, errwidth=0.6, order=result["Method"])
    handles, labels = ax.get_legend_handles_labels()
    try:
        index_of = labels.index("Dataset")
        handles.pop(index_of)
        labels.pop(index_of)
    except ValueError:
        pass
    ax.legend_.remove()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc="best")  # , bbox_to_anchor=(1,0.7))

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    plt.tight_layout()
    save_plot(Path(path_list[0]).parent, "online_bar_plot_new")
    print("saved plot to {}".format(Path(path_list[0]).parent))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("experiment_base_dir", nargs="+")
    p.add_argument("--reward", choices=["item", "list"], default="item")
    args = p.parse_args()

    base_paths = args.experiment_base_dir

    # online_eval_bar_plot(base_paths, args.reward)

    base_path = Path(base_paths[0])

    # print("Creating the train reward plot")
    # df_train = load_train_df(base_path, max_episode_num=2_000)
    # # metrics_progress_plot(df_train, base_path)
    # # # print("Creating q-value -return plot ")
    # train_return_q_value_plot(df_train, base_path)
    #
    # # print("Loading the valid files")
    valid_df = load_metrics_to_df(base_path, max_episode_limit=5_000)
    metrics_progress_plot(valid_df, base_path, y="HitRate", df_type="valid", sliding_window=10)

    # print("Creating plots based on the experiment")
    # for y in ["Hitrate"]:
    #     name, lgd = direct_factor_comparison(valid_df, factor="Experiment", y=y)
    #     save_plot(base_path, name, bbox_extra_artists=(lgd,))
    #     plt.show()
    #
    # print("Creating double-figures with return and and diversity")
    # for exp in valid_df.Experiment.unique():
    #     for d in ["ML-1M", "ML-25M"]:
    #         val = create_metrics_plot(valid_df, exp, d)
    #         if val:
    #             save_plot(base_path, val[0])
    # del valid_df
    # print("Loading the test metrics and creating a mean over seeds table")
    # test_df = load_metrics_to_df(base_path, file_name="test_metrics")
