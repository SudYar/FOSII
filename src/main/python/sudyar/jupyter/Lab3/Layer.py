import numpy as np


class Layer:
    """
    Основной класс слоев.
    """
    input_signal: np.ndarray
    output_signal: np.ndarray

    def forward(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, error_signal: np.ndarray, is_need_out) -> np.ndarray | None:
        raise NotImplementedError

    def update_params(self, learning_rate: float):
        raise NotImplementedError