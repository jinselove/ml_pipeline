import matplotlib.pyplot as plt


def generate_X_Y(df,
                 sample_frac=1,
                 target_cols=None,
                 drop_cols=None,
                 is_classification=False,
                 class_boundary=0,
                 random_state=42,
                 savefolder="results/",
                 verbose=True):
    """ generate X, Y for model building

    :param df: processed dataframe
    :param target_cols: target columns (could be multilabel classification or multitarget regression model)
    :param drop_cols: columns to drop
    :param is_classification: by default False. If it is true, then need to convert Y to binary using class_boundary
    :param class_boundary: default boundary to classify if tire is worn out or not
    :param random_state: random state to sample data
    :param savefolder: folder to save results
    :param verbose: verbose level
    """
    df = df.sample(frac=sample_frac, random_state=random_state)
    if is_classification:
        Y = (df[target_cols].values > class_boundary) * 1
    else:
        Y = df[target_cols].values

    try:
        df = df.drop(target_cols, axis=1)
        if drop_cols:
            df = df.drop(drop_cols, axis=1)
        X = df.values
        feature_names = df.columns.tolist()
    except Exception as ep:
        print("Warning: {}".format(ep))

    if verbose:
        print("*" * 40 + "Data report" + "*" * 40)
        print("X.shape = {}".format(X.shape))
        print("Y.shape = {}".format(Y.shape))
        print("Predictors: {}".format(df.columns))
        print("Target: {}".format(target_cols))
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(Y)
        ax.set_title("target_distribution")
        f.savefig(savefolder + "target_distribution.png")
    return X, Y, feature_names


def cal_correlation(df, cols=None):
    """analyze correlation for data"""
    corr = df[cols].astype(float).corr()
    return corr
