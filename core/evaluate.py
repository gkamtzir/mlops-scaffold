from typing import Union, List
from sklearn.metrics import mean_squared_error, mean_absolute_error,\
    mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from core.NumpyArrayEncoder import NumpyArrayEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import mlflow
from pathlib import Path


def evaluate(model_results: Union[GridSearchCV, BayesSearchCV, RandomizedSearchCV], x_test: pd.DataFrame,
             y_test: pd.DataFrame, metadata: any, features, model_type: str, experiment_path: str):
    """
    Evaluates and stores the results to JSON file in the given folder.
    :param model_results: The `GridSearchCV`, `BayesSearchCV` or the `RandomizedSearchCV` results.
    :param x_test: The testing features.
    :param y_test: The testing targets.
    :param metadata: Any extra metadata to be saved.
    :param features: The available features.
    :param model_type: The model type.
    :param experiment_path: The experiment path.
    """
    y_prediction = model_results.predict(x_test)
    results = {
        "mean_absolute_error": mean_absolute_error(y_test, y_prediction),
        "mean_squared_error": mean_squared_error(y_test, y_prediction),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y_test, y_prediction),
        "r2": r2_score(y_test, y_prediction),
        "best_score": model_results.best_score_,
        "best_params": model_results.best_params_,
        "cv_results": model_results.cv_results_
    }

    # Calculating feature importance.
    feature_importance(model_results.best_estimator_, features, x_test, y_test, model_type, experiment_path)

    # Calculating partial dependence plots.
    for index, column in enumerate(features.tolist()):
        partial_dependence(model_results.best_estimator_, x_test, index, column, model_type, experiment_path)

    # Visualizing the predictions.
    visualize_predictions(y_test, y_prediction, model_type, experiment_path)

    mlflow.log_metric("test_mean_absolute_error", mean_absolute_error(y_test, y_prediction))
    mlflow.log_metric("test_mean_squared_error", mean_squared_error(y_test, y_prediction))
    mlflow.log_metric("test_mean_absolute_percentage_error", mean_absolute_percentage_error(y_test, y_prediction))
    mlflow.log_metric("test_r2_score", r2_score(y_test, y_prediction))

    for i in range(len(model_results.cv_results_["params"])):
        mlflow.log_metric("mean_fit_time", model_results.cv_results_["mean_fit_time"][i])
        mlflow.log_metric("std_fit_time", model_results.cv_results_["std_fit_time"][i])
        mlflow.log_metric("std_score_time", model_results.cv_results_["std_score_time"][i])
        mlflow.log_metric("mean_test_score", model_results.cv_results_["mean_test_score"][i])
        mlflow.log_metric("std_test_score", model_results.cv_results_["std_test_score"][i])
        mlflow.log_metric("rank_test_score", model_results.cv_results_["rank_test_score"][i])

    with open(f"./results/{model_type}/{experiment_path}/results.json", "w", encoding="utf8") as file:
        json.dump({**results, **metadata}, file, cls=NumpyArrayEncoder)


def feature_importance(model: BaseSearchCV, features: pd.Index, x_test: pd.DataFrame, y_test: pd.DataFrame,
                       model_type: str, experiment_path: str):
    """
    Calculates and plots the feature importance and permutation
    importance of the given model.
    :param model: The model to be used.
    :param features: The available features.
    :param x_test: The testing features.
    :param y_test: The testing targets.
    :param model_type: The model type.
    :param experiment_path: The experiment path.
    """
    # Feature importance.
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.title("Feature Importance")

    # Permutation importance.
    result = permutation_importance(model, x_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=np.array(features)[sorted_idx])
    plt.title("Permutation Importance (on test set)")
    plt.xlabel("Decrease in accuracy")
    fig.tight_layout()

    fig.savefig(f"./results/{model_type}/{experiment_path}/feature_importance.png", dpi=200)


def partial_dependence(model: BaseSearchCV, x: pd.DataFrame, feature_index: int, feature_name: str, model_type: str,
                       experiment_path: str):
    """
    Plots the partial dependence plots for the given model and features.
    :param model: The model.
    :param x: The data.
    :param feature_index: The feature index to be plotted.
    :param feature_name: The feature name to be plotted.
    :param model_type: The regressor type.
    :param experiment_path: The experiment name.
    """
    PartialDependenceDisplay.from_estimator(model, x, features=[feature_index], kind="average")

    figure = plt.gcf()
    figure.suptitle("Partial Dependence Plot")
    figure.savefig(f"./results/{model_type}/{experiment_path}/partial_dependence_{feature_name}.png", dpi=200)


def visualize_predictions(y_test: pd.DataFrame, y_prediction: pd.DataFrame, model_type: str,
                          experiment_path: str):
    """
    Visualizing the predictions.
    :param y_test: The actual targets.
    :param y_prediction: The predicted targets.
    :param model_type: The model type.
    :param experiment_path: The experiment path.
    """
    figure = plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_prediction)
    plt.title("Visualizing Predictions")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")

    # Perfect predictions.
    plt.plot(y_test, y_test, "r")

    figure.tight_layout()
    figure.savefig(f"./results/{model_type}/{experiment_path}/predictions.png", dpi=200)


def feature_correlation(df: pd.DataFrame, name: str):
    """
    Plots the correlation between the features of the given dataframe.
    :param df: The dataframe.
    :param name: The file name.
    """
    Path(f"./results/general").mkdir(parents=True, exist_ok=True)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(df.corr(), interpolation='nearest')
    figure.colorbar(cax)

    ax.set_xticklabels([''] + df.columns.tolist())
    ax.set_yticklabels([''] + df.columns.tolist())

    figure.savefig(f"./results/general/{name}.png", dpi=200)

