from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def standard_scale(x_train, x_test):
    """
    Uses standard scaler on the given data.
    :param x_train: The training data.
    :param x_test: The testing data.
    :return: The scaled training and testing data.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.astype(np.float))
    x_test = scaler.transform(x_test.astype(np.float))
    return x_train, x_test


def minmax_scale(x_train, x_test):
    """
    Uses min-max scaler on the given data.
    :param x_train: The training data.
    :param x_test: The testing data.
    :return: The scaled training and testing data.
    """
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.astype(np.float))
    x_test = scaler.transform(x_test.astype(np.float))
    return x_train, x_test
