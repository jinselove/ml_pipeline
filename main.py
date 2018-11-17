import pandas as pd
import numpy as np
from data_handler import generate_X_Y
from model_pipeline import *
import xgboost as xgb
import pickle
import datetime
import os
import multiprocessing

if __name__ == '__main__':
    n_cpus = multiprocessing.cpu_count()
    # read data
    data_file = "../raw_data/df_01:11:51.911271.csv"
    print("Reading data from {0}".format(data_file))
    df = pd.read_csv("../raw_data/df_01:11:51.911271.csv")
    #
    # ######################
    # # classifier
    # ######################
    #
    # # create folder to store results
    # savefolder = "results_" + str(datetime.datetime.now()).replace(" ", "-") + "/"
    # os.mkdir(savefolder)
    # print("Results will be saved to {0}".format(savefolder))
    #
    # # generate X, Y
    # print("Generating X, Y, feature_names for training")
    # X, Y, feature_names = generate_X_Y(df=df,
    #                                    sample_frac=0.001,
    #                                    target_cols="tip_percentage",
    #                                    drop_cols=["tip_amount", "total_amount"],
    #                                    is_classification=True,
    #                                    class_boundary=0,
    #                                    savefolder=savefolder,
    #                                    random_state=42)
    #
    # clf = xgb.XGBClassifier()
    #
    # search_params = {
    #     'min_child_weight': np.arange(1, 6, 2),
    #     'max_depth': np.arange(3, 10, 2),
    #     'gamma': np.arange(0, 0.5, 0.1),
    #     'subsample': np.arange(0.2, 1.0, 0.2),
    #     'colsample_bytree': np.arange(0.2, 1, 0.2),
    #     'learning_rate': np.arange(0.01, 0.1, 0.02),
    #     'n_estimators': np.arange(50, 300, 50)
    # }
    #
    # print("Train model on {} cpus".format(n_cpus))
    # cv_clf = clf_RandomSearchCV_pipeline(clf, X, Y, feature_names, test_size=0.2,
    #                                      savefolder=savefolder,
    #                                      search_params=search_params, n_jobs=n_cpus, n_iter=100)
    #
    # # save the model to disk
    # print("*"*40 + "Dumping the CV_clf" + "*"*40)
    # pickle.dump(cv_clf.best_estimator_, open(savefolder + "cv_xgb.dat", 'wb'))

    #######################
    ## Regressor
    #######################
    # create folder to store results
    savefolder = "results_" + str(datetime.datetime.now()).replace(" ", "-") + "/"
    os.mkdir(savefolder)
    print("Results will be saved to {0}".format(savefolder))

    # generate X, Y
    print("Generating X, Y, feature_names for training")
    X, Y, feature_names = generate_X_Y(df=df,
                                       sample_frac=0.001,
                                       target_cols="tip_percentage",
                                       drop_cols=["tip_amount", "total_amount"],
                                       is_classification=False,
                                       savefolder=savefolder,
                                       random_state=42)

    reg = xgb.XGBRegressor()

    search_params = {
        'min_child_weight': np.arange(1, 6, 2),
        'max_depth': np.arange(3, 10, 2),
        'gamma': np.arange(0, 0.5, 0.1),
        'subsample': np.arange(0.2, 1.0, 0.2),
        'colsample_bytree': np.arange(0.2, 1, 0.2),
        'learning_rate': np.arange(0.01, 0.1, 0.02),
        'n_estimators': np.arange(50, 300, 50)
    }

    print("Train model on {} cpus".format(n_cpus))
    cv_reg = reg_RandomSearchCV_pipeline(reg, X, Y, feature_names, test_size=0.2,
                                         savefolder=savefolder,
                                         search_params=search_params, n_jobs=n_cpus, n_iter=100)

    # save the model to disk
    print("*"*40 + "Dumping the CV_clf" + "*"*40)
    pickle.dump(cv_reg.best_estimator_, open(savefolder + "cv_xgb.dat", 'wb'))