import numpy as np

from Layer import Layer


class Flatten(Layer):
    output_signal: np.ndarray
    input_signal: np.ndarray
    input_size: int
    output_size: int

    def __init__(self):
        pass

    def forward(self, signal:np.ndarray):
        self.input_size = signal.shape
        b, c, h, w = signal.shape
        return signal.reshape(b, c * h * w)

    def backward(self, error_signal: np.ndarray, is_need_out=True):
        return error_signal.reshape(self.input_size)

    def update_params(self, learning_rate: float):
        pass
