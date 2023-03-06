import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from experiment import run_experiment, create_run_name
from utilities import read_experiment_parameters
from skopt.space import Categorical, Integer
import json


if __name__ == "__main__":
    # Reading experiment parameters.
    data_file, experiment_id, experiment_name, experiment_tags, mode, scale = read_experiment_parameters()

    df = pd.read_csv(data_file)

    print(f"Total number of rows: {df.shape}")

    y = df["Y"]
    X = df.drop("Y", axis=1)

    run_name = create_run_name("decision_trees", mode, scale)

    metadata = {
        "mode": mode,
        "scale": scale
    }

    experiment_details = {
        "id": experiment_id,
        "name": experiment_name,
        "artifact": data_file,
        "tags": json.loads(experiment_tags) if experiment_tags is not None else None
    }

    # Initializing model.
    model = DecisionTreeRegressor()

    # Setting up parameters.
    if mode == "bayesian":
        parameters = {
            "criterion": Categorical(["squared_error", "friedman_mse", "absolute_error", "poisson"]),
            "splitter": Categorical(["best", "random"]),
            "max_depth": Integer(2, 20),
            "min_samples_leaf": Integer(5, 80)
        }
    else:
        parameters = {
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter": ["best", "random"],
            "max_depth": [2, 3, 5, 7, 10, 15, 20],
            "min_samples_leaf": [5, 8, 10, 15, 40, 80]
        }

    # Run the experiment
    run_experiment(model, parameters, X, y, 0.2, "decision-trees",
                   run_name, experiment_details, metadata, mode, scale)
