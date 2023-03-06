from sklearn.preprocessing import StandardScaler
import numpy as np


def standard_scale(x_train, x_test):
    """
    Uses standard scaler on the given data.
    :param x_train: The training data.
    :param x_test: The testing data.
    :return: The scaled training and testing data.
    """
    s_scaler = StandardScaler()
    x_train = s_scaler.fit_transform(x_train.astype(np.float))
    x_test = s_scaler.transform(x_test.astype(np.float))
    return x_train, x_test

