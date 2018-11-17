from data_handler import *
from visualization_handler import *
from performance_handler import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def clf_RandomSearchCV_pipeline(clf,
                                X, Y, feature_names,
                                test_size=0.2,
                                search_params=None,
                                folds=5,
                                random_state=42,
                                n_jobs=1,
                                n_iter=10,
                                savefolder="results/",
                                verbose=1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    # train model
    cv_clf = RandomizedSearchCV(clf,
                                param_distributions=search_params,
                                scoring='roc_auc',
                                n_jobs=n_jobs,
                                cv=folds,
                                verbose=verbose,
                                n_iter=n_iter,
                                random_state=random_state)
    print("*" * 40 + "start training" + "*" * 40)
    start_time = timer(None)
    cv_clf.fit(X_train, Y_train)
    timer(start_time)
    print("*" * 40 + "save Grid search result" + "*" * 40)
    report(cv_clf.cv_results_, savefolder + "classifier_cv_results.csv")
    # visualization
    print("*" * 40 + "making predictions" + "*" * 40)
    Y_pred = cv_clf.predict(X_test)
    Y_pred_prob = cv_clf.predict_proba(X_test)
    print("*" * 40 + "model performance on test data" + "*" * 40)
    classification_performance_report(Y_test, Y_pred,
                                      Y_pred_prob,
                                      savefolder + "classification_report.json")
    print("*" * 40 + "plot feature importance" + "*" * 40)
    feature_importance_plot(features=feature_names,
                            feature_importances=cv_clf.best_estimator_.feature_importances_,
                            savefolder=savefolder,
                            kind="bar")

    return cv_clf


def clf_GridSearchCV_pipeline(clf,
                              X, Y, feature_names,
                              test_size=0.2,
                              search_params=None,
                              folds=5,
                              random_state=42,
                              savefolder="results/",
                              n_jobs=1,
                              verbose=1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    # train model
    cv_clf = GridSearchCV(clf,
                          param_grid=search_params,
                          scoring='roc_auc',
                          n_jobs=n_jobs,
                          cv=folds,
                          verbose=verbose)
    print("*" * 40 + "start training" + "*" * 40)
    start_time = timer(None)
    cv_clf.fit(X_train, Y_train)
    timer(start_time)
    print("*" * 40 + "save Grid search result" + "*" * 40)
    report(cv_clf.cv_results_, savefolder + "classifier_cv_results.csv")
    # visualization
    print("*" * 40 + "making predictions" + "*" * 40)
    Y_pred = cv_clf.predict(X_test)
    Y_pred_prob = cv_clf.predict_proba(X_test)
    print("*" * 40 + "model performance on test data" + "*" * 40)
    classification_performance_report(Y_test, Y_pred,
                                      Y_pred_prob,
                                      savefolder + "classification_report.json")
    print("*" * 40 + "plot feature importance" + "*" * 40)
    feature_importance_plot(features=feature_names,
                            feature_importances=cv_clf.best_estimator_.feature_importances_,
                            savefolder=savefolder,
                            kind="bar")

    return cv_clf


def reg_RandomSearchCV_pipeline(reg, X, Y, feature_names,
                                test_size=0.2,
                                search_params=None,
                                folds=5,
                                random_state=42,
                                n_jobs=1,
                                n_iter=10,
                                savefolder="results/",
                                verbose=1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    # train model
    cv_reg = RandomizedSearchCV(reg,
                                param_distributions=search_params,
                                n_jobs=n_jobs,
                                cv=folds,
                                verbose=verbose,
                                n_iter=n_iter,
                                random_state=random_state)
    print("*" * 40 + "start training" + "*" * 40)
    start_time = timer(None)
    cv_reg.fit(X_train, Y_train)
    timer(start_time)
    print("*" * 40 + "Save Grid search result" + "*" * 40)
    report(cv_reg.cv_results_, savefolder + "xgb_regressor_cv_results.csv")
    # visualization
    print("*" * 40 + "making predictions" + "*" * 40)
    Y_pred = cv_reg.predict(X_test)
    print("*" * 40 + "model performance on test data" + "*" * 40)
    regression_performance_report(Y_test, Y_pred, savefolder + "regression_report.json")
    print("*" * 40 + "plot feature importance" + "*" * 40)
    feature_importance_plot(features=feature_names,
                            feature_importances=cv_reg.best_estimator_.feature_importances_,
                            savefolder=savefolder,
                            kind="bar")
    return cv_reg


def reg_GridSearchCV_pipeline(reg, X, Y, feature_names,
                              test_size=0.2,
                              search_params=None,
                              folds=5,
                              random_state=42,
                              n_jobs=1,
                              savefolder="results/",
                              verbose=1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    # train model
    cv_reg = GridSearchCV(reg,
                          param_grid=search_params,
                          n_jobs=n_jobs,
                          cv=folds,
                          verbose=verbose)
    print("*" * 40 + "start training" + "*" * 40)
    start_time = timer(None)
    cv_reg.fit(X_train, Y_train)
    timer(start_time)
    print("*" * 40 + "Save Grid search result" + "*" * 40)
    report(cv_reg.cv_results_, savefolder + "xgb_regressor_cv_results.csv")
    # visualization
    print("*" * 40 + "making predictions" + "*" * 40)
    Y_pred = cv_reg.predict(X_test)
    print("*" * 40 + "model performance on test data" + "*" * 40)
    regression_performance_report(Y_test, Y_pred, savefolder + "regression_report.json")
    print("*" * 40 + "plot feature importance" + "*" * 40)
    feature_importance_plot(features=feature_names,
                            feature_importances=cv_reg.best_estimator_.feature_importances_,
                            savefolder=savefolder,
                            kind="bar")
    return cv_reg
