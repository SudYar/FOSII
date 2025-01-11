import numpy as np
import enum

rng = np.random.default_rng(51)


def relu_f(x: np.ndarray | float):
    return np.maximum(0, x)


def relu_df(x: np.ndarray | float):
    return np.where(x > 0, 1, 0)


def softplus_f(x: np.ndarray | float):
    return np.log(np.exp(x) + 1)


def softplus_df(x: np.ndarray | float):
    """Принимает на вход результат softplus_f (аналогично сигмоиде и т.д.)"""
    return 1 - np.exp(-x)


def softmax_f(x: np.ndarray | float):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def sigmoid_f(x: np.ndarray | float):
    return 1 / (1 + np.exp(-x))


def sigmoid_df(x: np.ndarray | float):
    return x * (1. - x)


def tanh_f(x: np.ndarray | float):
    return np.tanh(x)


def tanh_df(x: np.ndarray | float):
    return 1. - np.square(x)


def mse(y_pred: np.array, y: np.array):
    return np.mean(np.square(y_pred - y))


def dmse(y_pred, y):
    n = y.shape[0]
    return (2 / n) * (y_pred - y)


def mae(y_pred: np.array, y: np.array):
    return np.mean(np.abs(y_pred - y))

def dmae(y_pred, y):
    batch_size = y.shape[0]
    return np.where(y_pred > y, 1, -1) / batch_size


class Functions(enum.Enum):

    relu = (relu_f, relu_df)
    softplus = (softplus_f, softplus_df)
    sigmoid = (sigmoid_f, sigmoid_df)
    tanh = (tanh_f, tanh_df)
    none = (lambda x: x, lambda x: 1)
    softmax = (softmax_f, None)

    def __init__(self, act_f, act_df):
        self.act_f = act_f
        self.act_df = act_df

    def calc(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.act_f(x)

    def derivative(self, x: np.ndarray | float) -> np.ndarray | float:
        if self.act_df is None:
            raise NotImplementedError(case_info=f"Для функции '{self.act_f.name}' не предусмотрена производная")
        else:
            return self.act_df(x)


def initialize_weights(in_size: int | tuple, size: int | tuple, out_size: int | tuple = None, method='he') -> np.ndarray:
    """

    :param in_size: Размерность входящего сигнала
    :param size: Размерность в каком выдать инициализированные веса
    :param out_size: Размерность выходящего сигнала (обязательно для метода Ксавье)
    :param method: Метод Глора ='he' - инициализации для слоев с ReLu. Иначле используется метод Ксавье -
        для сигмоид и танценца
    :return:
    """
    fan_in = np.sum(in_size)
    if out_size:
        fan_out = np.sum(out_size)

    if method == 'he':
        limit = np.sqrt(6 / fan_in)
    else:
        limit = np.sqrt(6 / (fan_in + fan_out))

    return np.array(rng.uniform(-limit, limit + 1e-5,
                                size=size))
