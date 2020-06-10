from matplotlib.ticker import ScalarFormatter

from figures.visualize_results import *


# plt.style.use("seaborn-paper")
# mpl.rcParams.update(nice_fonts)


def intro_rec_trend_plot():
    df = pd.read_csv("/home/alex/Dokumente/write_thesis/documents/recommender-trend.csv", delim_whitespace=True)
    df.columns = ["Year", "reinforcement learning recommendation", "recommendation system"]
    df = df.loc[df.Year < 2020]
    # df[df == 0] = 1
    df = df.melt('Year', var_name='Search term', value_name='Publications')
    g = sns.lineplot(x="Year", y="Publications", hue='Search term', data=df)
    g.set(yscale="symlog")
    g.yaxis.set_major_formatter(ScalarFormatter())
    g.legend(loc="lower left", bbox_to_anchor=(0, 0.5))
    plt.ylabel("New Publications")
    # plt.show()
    save_plot(Path("/home/alex/Dokumente/write_thesis/figures/intro"), "rec-trend")


if __name__ == '__main__':
    intro_rec_trend_plot()
