import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from core.train import train
from pathlib import Path
from core.evaluate import evaluate
from core.scale import standard_scale, minmax_scale
from typing import Any
from sklearn.tree import DecisionTreeRegressor
import mlflow
import time


def run_experiment(model: DecisionTreeRegressor, parameters: Any, x: pd.DataFrame, y: pd.DataFrame,
                   test_size: float, model_type: str, run_name: str, experiment_details: Any, metadata: Any,
                   mode="grid_search", scale=None):
    """
    Run experiment based on the given data.
    :param model: The model to be trained.
    :param parameters: The parameters to be used.
    :param x: All available features.
    :param y: All available targets.
    :param test_size: The test size to be used.
    :param model_type: The model type.
    :param run_name: The name of the run.
    :param experiment_details: The experiment details.
    :param metadata: The metadata.
    :param mode: (Optional) The hyperparameter optimization method.
    :param scale: (Optional) Indicates the method based on which
    the data will be scaled. Available options are `standard` and `minmax`.
    """
    # Create experiment.
    experiment_id = None
    if experiment_details["id"] is not None:
        new_experiment = mlflow.get_experiment(experiment_details["id"])
        if new_experiment is not None:
            experiment_id = new_experiment.experiment_id
        else:
            return
    else:
        experiment_id = mlflow.create_experiment(experiment_details["name"],
                                                 tags=experiment_details["tags"])
    # Creating experiment folder.
    Path(f"./results/{model_type}/{experiment_id}/{run_name}").mkdir(parents=True, exist_ok=True)

    mlflow.autolog()
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Log parameters.
        mlflow.log_artifact(experiment_details["artifact"])
        mlflow.log_param("Mode", metadata["mode"])
        mlflow.log_param("Scale", metadata["scale"])
        mlflow.log_param("features", ', '.join(x.columns.tolist()))

        # Splitting training and test set.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

        if scale == "standard":
            x_train, x_test = standard_scale(x_train, x_test)
        elif scale == "minmax":
            x_train, x_test = minmax_scale(x_train, x_test)

        mlflow.set_tag("Model", model_type)

        mlflow.log_param("test_size", test_size)

        # Training model.
        model_results = train(model, parameters, x_train, y_train, mode=mode)

        metadata = {
            "test_size": test_size,
        }

        # Evaluating and storing results.
        evaluate(model_results, x_test, y_test, metadata, x.columns, model_type, f"{experiment_id}/{run_name}")

        # Saving the model.
        dump(model_results.best_estimator_, f"./results/{model_type}/{experiment_id}/{run_name}/model.joblib")
        mlflow.sklearn.log_model(model_results.best_estimator_, "model")


def create_run_name(prefix: str, mode: str, scale: str) -> str:
    """
    Creates run name based on the given parameters.
    :param prefix: The prefix to be used.
    :param mode: The hyperparameter optimization mode.
    :param scale: The scale mode.
    :return: The run name.
    """
    return f"{prefix}_mode_{mode}_scale_{scale}_{int(time.time())}"
