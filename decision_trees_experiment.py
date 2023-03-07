from sklearn.tree import DecisionTreeRegressor
from core.experiment import run_experiment
from skopt.space import Categorical, Integer


if __name__ == "__main__":
    # Initializing model.
    model = DecisionTreeRegressor()

    # Setting up parameters.
    # Use these for Bayesian optimization.
    parameters = {
        "criterion": Categorical(["squared_error", "friedman_mse", "absolute_error", "poisson"]),
        "splitter": Categorical(["best", "random"]),
        "max_depth": Integer(2, 20),
        "min_samples_leaf": Integer(5, 80)
    }
    # Use these for Grid Search optimization.
    parameters = {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        "splitter": ["best", "random"],
        "max_depth": [2, 3, 5, 7, 10, 15, 20],
        "min_samples_leaf": [5, 8, 10, 15, 40, 80]
    }

    # Run the experiment
    run_experiment(model, parameters)
