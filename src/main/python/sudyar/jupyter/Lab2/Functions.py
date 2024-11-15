from typing import Literal

import numpy as np
import enum

rng = np.random.default_rng(51)


def relu_f(x: np.ndarray | float):
    return np.maximum(0, x)


def relu_df(x: np.ndarray | float):
    return np.where(x > 0, 1, 0)


def softmax_f(x: np.ndarray | float):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def sigmoid_f(x: np.ndarray | float):
    return 1 / (1 + np.exp(-x))


def sigmoid_df(x: np.ndarray | float):
    return sigmoid_f(x) * (1 - sigmoid_f(x))


def tanh_f(x: np.ndarray | float):
    return np.tanh(x)


def tanh_df(x: np.ndarray | float):
    return 1 - np.square(tanh_f(x))


def mse(y_pred: np.array, y: np.array):
    return np.mean(np.square(y_pred - y))


def dmse(y_pred, y):
    n = y.shape[0]
    return (2 / n) * (y_pred - y)


class Functions(enum.Enum):

    relu = (relu_f, relu_df)
    sigmoid = (sigmoid_f, sigmoid_df)
    tanh = (tanh_f, tanh_df)
    none = (lambda x: x, lambda x: 1)
    softmax = (softmax_f, None)

    def __init__(self, act_f, act_df):
        self.act_f = act_f
        self.act_df = act_df

    def calc(self, x: np.ndarray | float):
        return self.act_f(x)

    def derivative(self, x: np.ndarray | float):
        if self.act_df is None:
            raise NotImplementedError(case_info=f"Для функции '{self.act_f.name}' не предусмотрена производная")
        else:
            return self.act_df(x)
