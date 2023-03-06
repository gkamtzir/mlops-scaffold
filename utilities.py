import argparse


def read_experiment_parameters():
    """
    Reads the experiment parameters from the command line and
    returns them in the appropriate format.
    :return: The experiment parameters.
    """
    parser = argparse.ArgumentParser()

    # Setting arguments.
    # INFO: You can include as many arguments as you like depending
    # on your project.
    parser.add_argument("--experiment-id", help="The experiment id", default=None)
    parser.add_argument("--experiment-name", help="The experiment name", default=None)
    parser.add_argument("--experiment-tags", help="The experiment tags", type=str, default=None)
    parser.add_argument("--mode", help="The hyperparameter optimization method", default="grid_search")

    args = parser.parse_args()
    return args.experiment_id, args.experiment_name, args.experiment_tags, args.mode
