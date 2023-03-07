import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from core.train import train
from pathlib import Path
from core.evaluate import evaluate
from core.scale import standard_scale, minmax_scale
from typing import Any
from sklearn.tree import DecisionTreeRegressor
from core.utilities import read_experiment_parameters
import mlflow
import time
import json


def run_experiment(model: DecisionTreeRegressor, parameters: Any):
    """
    Run experiment based on the given data.
    :param model: The model to be trained.
    :param parameters: The parameters to be used.
    the data will be scaled. Available options are `standard` and `minmax`.
    """
    # Reading input.
    data_file, experiment_id, experiment_name, experiment_tags, \
        mode, scale, test_size = read_experiment_parameters()
    df = pd.read_csv(data_file)

    # Create experiment.
    run_name = create_run_name(experiment_name, mode, scale)

    if experiment_id is not None:
        new_experiment = mlflow.get_experiment(experiment_id)
        if new_experiment is not None:
            experiment_id = new_experiment.experiment_id
            experiment_name = new_experiment.name
        else:
            return
    else:
        experiment_id = mlflow.create_experiment(experiment_name,
                                                 tags=experiment_tags)

    experiment_details = {
        "id": experiment_id,
        "name": experiment_name,
        "artifact": data_file,
        "tags": json.loads(experiment_tags) if experiment_tags is not None else None
    }

    print(f"Total number of rows: {df.shape}")

    y = df["Y"]
    X = df.drop("Y", axis=1)

    # Creating experiment folder.
    Path(f"./results/{experiment_name}/{run_name}").mkdir(parents=True, exist_ok=True)

    mlflow.autolog()
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Log parameters.
        mlflow.log_artifact(experiment_details["artifact"])
        mlflow.log_param("Mode", mode)
        mlflow.log_param("Scale", scale)
        mlflow.log_param("features", ', '.join(X.columns.tolist()))

        # Splitting training and test set.
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

        if scale == "standard":
            x_train, x_test = standard_scale(x_train, x_test)
        elif scale == "minmax":
            x_train, x_test = minmax_scale(x_train, x_test)

        mlflow.set_tag("Model", type(model).__name__)

        mlflow.log_param("test_size", test_size)

        # Training model.
        model_results = train(model, parameters, x_train, y_train, mode=mode)

        metadata = {
            "test_size": test_size,
        }

        # Evaluating and storing results.
        evaluate(model_results, x_test, y_test, metadata, X.columns, experiment_name, f"{run_name}")

        # Saving the model.
        dump(model_results.best_estimator_, f"./results/{experiment_name}/{run_name}/model.joblib")
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
