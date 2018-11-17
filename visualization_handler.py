import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def correlation_plot(df_, cols_=None, savefolder="results/"):
    """analyze correlation for data"""
    corr = df_[cols_].astype(float).corr()
    colormap = plt.cm.RdBu
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(corr, linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True, ax=ax)
    f.savefig(savefolder + "correlation_plot.png")


def pair_plot(df_, cols_=None, hue_col_=None, savefolder="results/"):
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.pairplot(df_[cols_], hue=hue_col_, palette='seismic', size=1.2, diag_kind='kde',
                 diag_kws=dict(shade=True), plot_kws=dict(s=10), ax=ax)
    ax.set_xticklabels([])
    ax.set_title("pair plot")
    f.savefig(savefolder + "pair_plot.png")


def distribution_plot(df_, cols_=None, savefolder="results/"):
    for col in cols_:
        print("*" * 40 + col)
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.distplot(df_[col], ax=ax)
        ax.set_title("distribution plot")
        f.savefig(savefolder + "distribution_plot.png")


def feature_importance_plot(features, feature_importances, savefolder="results/", kind="bar"):
    if kind == "bar":
        tmp = pd.DataFrame(data=feature_importances, index=features, columns=["importance"]).sort_values(
            by="importance")
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        tmp.plot(kind="bar", ax=ax)
        f.set_tight_layout(True)
        f.savefig(savefolder + "feature_importance_plot.png")
    else:
        print("kind: {0} not defined".format(kind))
