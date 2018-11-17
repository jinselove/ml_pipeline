from operator import itemgetter
from datetime import datetime
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
import numpy as np
import pandas as pd
import pickle
import json
import os


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Utility function to report best scores
def grid_search_report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def classification_performance_report(Y_test, Y_pred, Y_pred_prob, savefile):
    accuracy_score_ = accuracy_score(Y_test, Y_pred)
    roc_auc_score_ = roc_auc_score(Y_test, Y_pred_prob[:, 1])
    report_dict = {"accuracy_score": accuracy_score_,
                   "roc_auc_score": roc_auc_score_,
                   "classification_report": classification_report(Y_test, Y_pred)}

    print("*" * 40 + "classfication report" + "*" * 40)
    for key, value in report_dict.items():
        print("{0} : {1}".format(key, value))

    with open(savefile, 'w') as file:
        file.write(json.dumps(report_dict))


def regression_performance_report(y_true_, y_pred_, savefile):
    r2_score_ = r2_score(y_true=y_true_, y_pred=y_pred_)
    explained_variance_score_ = explained_variance_score(y_true=y_true_, y_pred=y_pred_)
    mean_squared_error_ = mean_squared_error(y_true=y_true_, y_pred=y_pred_)
    report_dict = {"r2_score": r2_score_,
                   "explained_variance_score": explained_variance_score_,
                   "mean_square_error": mean_squared_error_}

    print("*" * 40 + "regression report" + "*" * 40)
    for key, value in report_dict.items():
        print("{0} : {1}".format(key, value))

    with open(savefile, 'w') as file:
        file.write(json.dumps(report_dict))


def read_performance_report(report_dict_file):
    if os.path.isfile(report_dict_file):
        return json.loads(report_dict_file)
    print("{} does not exist".format(report_dict_file))
    return None


def report(cv_results, savefile):
    save_cv_results = {"params": cv_results["params"],
                       "rank_test_score": cv_results["rank_test_score"],
                       "mean_test_score": cv_results["mean_test_score"],
                       "std_test_score": cv_results["std_test_score"]}

    df_cv_results = pd.DataFrame(save_cv_results)
    df_cv_results.sort_values(by="rank_test_score", inplace=True)
    df_cv_results.to_csv(savefile, index=None)


def select_low_importance_feature(feature_names_, feature_importances_, threshold_=0.01):
    """return a list of feature names that has feature_importances smaller than threthold
    :param feature_names_: a list of feature_names
    :param feature_importances: a list/array of feature importances (which has the same length as feature_names)
    :param threshold_: threshold of selection
    """
    return np.array(feature_names_)[np.where(feature_importances_ < threshold_)[0]].tolist()
