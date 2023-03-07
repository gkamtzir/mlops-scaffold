# Machine Learning Ops. Project Scaffold
This project contains the scaffold for MLOps of a machine learning project.
MLOps are essential to most projects, starting from enterprise solutions and research
programs all the way to small scale or even learning projects. I've found myself
writing over and over again the same boilerplate for my machine learning and
data science projects, so I've decided to gather the required infrastructure
in the current repository in order to be able to bootstrap future projects with
ease. Of course, the project can be used as is or enhanced by anyone.

## Installation Requirements
All dependencies are declared in the `setup.py` file. For these, I strongly
recommend the use of a virtual environment. To install the dependencies run:
```bash
pip install .
```

### Basic Dependencies
- [pandas](https://github.com/pandas-dev/pandas/) is used for data manipulation.
- [numpy](https://github.com/numpy/numpy) is used for numerical operations.
- [matplotlib](https://github.com/matplotlib/matplotlib) is used to visualize the results.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) is used for the ready-to-use ML models.
- [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) is used for Bayesian optimization algorithms.
- [mlflow](https://github.com/mlflow/mlflow/) is used to handle the MLOps.

### Data Dependencies
One major difference between managing conventional software and machine learning
code is the data. The data used in a project are not static, but they do change
frequently. We try out different preprocessing methods and techniques searching,
for instance, performance improvements. This means that we need to additionally
version our data. The data size varies from a few KBs to GBs and TBs in cases.
That's why we utilize [DVC](https://github.com/iterative/dvc) which is a data
version control toolkit that runs on top of git. So, be sure to install DVC on your
machine. To fully incorporate DVC you will also need a remote storage. There are
plenty of [solutions](https://dvc.org/doc/user-guide/data-management/remote-storage)
depending on the project needs and the resources you have available. For smaller
projects I tend to use my Google Drive because it's really easy to set it up and
running in a couple of minutes.

## Project Structure and General Information
The data are stored in the `data` folder, any data related to MLFlow are stored
in the `mlruns` folder, while we manually keep some experiment results in the
`results` folder. All three folders are versioned via DVC.

Regarding the core scripts we have:
- `/core/utilities.py`: contains a utility that reads input parameters through the command line.
- `/core/scale.py`: contains the basic scaling functionalities.
- `/core/experiment.py`: contains the core management of the experiments.
- `/core/train.py`: handles the actual training of the models.
- `/core/evaluate.py`: contains basic model evaluation techniques such as error metrics.

All scripts contain some basic functionality that I frequenty use. The code is
structured in a way to make any enhancements or additions easy to be integrated.
The error metrics as well as some of the interpretability tools are used for
regression problems. However, they can be easily switched to the corresponding
classification metrics if needed.

The hyperparameter tuning can be done in three different ways out-of-the box using
K-fold Cross-Validation:
1) Grid Search
2) Random Search
3) Bayesian Optimization

## Running Experiments
Regardless of the model that is to be trained the core scripts remain the same. To
train a new model one would need to create a script similar to the `decision_trees_experiment.py`
that I have created to showcase how to run experiments. To run a new experiment execute:
```bash
py .\decision_trees_experiment.py --data-file "./data/diabetes.csv" --mode "grid_search"  --experiment-name "Test"
```
To include an additional run in the same experiment execute:
```bash
py .\decision_trees_experiment.py --data-file "./data/diabetes.csv" --mode "bayesian"  --experiment-id {experiment_id}
```

The details list of parameters can be found below:
- `--data-file {data_file}` The data file to be used in the training process.
- `--mode {hyperparameter_tuning_id}` `'grid_search'` -> Grid Search, `'bayesian'` -> Baysian, else -> Random.
- `--scale {scale}` `'standard'` -> Standard Scaler, `'minmax'` -> Min-Max Scaler.
- `--experiment-name {name}` The name of the experiment to be created.
- `--experiment-tags {tags}` A dictionary containing any extra information.
- `--experiment-id {id}` The id of an already existing experiment.