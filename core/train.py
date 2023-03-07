from typing import Union
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


def train(estimator, parameters, x_train: pd.DataFrame, y_train: pd.DataFrame,
          scoring="neg_mean_absolute_error", cv=5, mode="grid_search")\
        -> Union[GridSearchCV, BayesSearchCV, RandomizedSearchCV]:
    """
    Trains the given estimator with the provided parameters and training data.
    :param estimator: The estimator to be trained.
    :param parameters: The parameters to be trained with.
    :param x_train: The training features.
    :param y_train: The training target.
    :param scoring: (Optional) The scoring function.
    :param cv: (Optional) The number of folds.
    :param mode: (Optional) Indicates the mode to be used between `GridSearch` and `RandomizedSearch`.
    :return: The `GridSearchCV`, `RandomizedSearchCV` or `BayesSearchCV` fitted on the given data.
    """
    if mode == "grid_search":
        search_cv = GridSearchCV(estimator=estimator, param_grid=parameters, scoring=scoring, cv=cv,
                                 return_train_score=True, n_jobs=-1)
    elif mode == "bayesian":
        search_cv = BayesSearchCV(estimator=estimator, search_spaces=parameters, scoring=scoring, cv=cv, n_iter=50,
                                  return_train_score=True, random_state=1, n_jobs=-1)
    else:
        search_cv = RandomizedSearchCV(estimator=estimator, param_grid=parameters, scoring=scoring,
                                       return_train_score=True, cv=cv, n_jobs=-1)

    search_cv.fit(x_train, y_train)

    return search_cv
