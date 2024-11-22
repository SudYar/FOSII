from typing import Optional

import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse


class RNNSimple(Layer):
    """
    Добавляет предыдущие состояния к сигналу перед Dense
    """
    # weights.shape = (input_size; output_size)
    weights: np.ndarray
    # bias.shape = (output_size)
    bias: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    input_signal: np.ndarray
    input_size: int
    seq_len: int
    output_size: int
    output_signal: np.ndarray

    def __init__(self, input_size: tuple[int, int], output_size: int, activation_function: Functions):
        self.activation_function = activation_function
        self.input_size = input_size[1]
        self.seq_len = input_size[0]
        self.output_size = output_size
        self.weights = self._init_weights(size=(self.input_size + self.output_size, self.output_size))
        self.bias = self._init_biases(self.output_size)

    def _init_weights(self, size: tuple):
        limit = np.sqrt(6. / (np.sum(size)))
        return np.array(rng.uniform(-limit, limit + 1e-5, size=size))

    def _init_biases(self, output_size):
        return np.array(rng.random(output_size) * 2 - 1)

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """

        :param signal: np.ndarray - Входящий сигнал, размерности (batch, sequence_length, output_size)
        :return: np.ndarray - выходящий сигнал размерности (batch, sequence_length, output_size
        """

        #         in_sig(batch, in), w (in, out), + b (out ,)
        self.output_signal = self.predict(signal)
        return self.output_signal

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if signal[0].shape != (self.seq_len, self.input_size):
            raise ValueError(f"Размерность сигнала{signal[0].shape} отличается от " +
                             f"размерности входящего слоя: {(self.seq_len, self.input_size)}")

        result = np.zeros((signal.shape[0], self.seq_len, self.output_size))
        summator = result.copy()
        self.input_signal = np.concatenate((signal, result), axis=-1)
        for t in range(self.seq_len):
            if t > 0:
                self.input_signal[:, t, self.input_size:] = result[:, t - 1]
            summator[:, t] = np.matmul(self.input_signal[:, t], self.weights) + self.bias
            result[:, t] = self.activation_function.calc(summator[:, t])
        return result

    def backward(self, error_signal: np.ndarray, is_need_out=True) -> np.ndarray | None:
        """

        :param error_signal: переданный дельта с предыдущих слоев (размерности (batch, sequence_len, out))
        :param is_need_out: Необходимость проталкивать delta дальше. Если это первый слой, то нет необходимости
        :return:
        """
        result = np.zeros_like(self.input_signal[:, :, :self.input_size])
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros(self.output_size)
        for t in reversed(range(self.seq_len)):
            # f (batch, out) * ((batch, out) + (batch, out) or 0)
            sum_error = error_signal[:, t] + (delta_t_plus_1 if t+1 < self.seq_len else 0)
            delta_t = self.activation_function.derivative(self.output_signal[:, t]) * (sum_error)
            # (batch, in+out).T x (batch, out) -> (in+out, out)
            self.dW += np.matmul(self.input_signal[:, t].T, delta_t)
            self.db += np.sum(delta_t, axis=0)
            # delta, чтобы передать для t-1;    # веса для скрытого слоя размерности V(out, out)
            delta_t_plus_1 = np.matmul(delta_t, self.weights[self.input_size:].T) if t != 0 else 0 #условие чтобы не считать при t = 0
            # [error(batch, out)x w(in, out).T  ] -> (batch, in)
            if is_need_out:
                result[:, t] = np.matmul(delta_t, self.weights[:self.input_size].T)
        return result if is_need_out else None

    def update_params(self, learning_rate: Optional[float]):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db


def test_forward():
    inp_len = 5
    out_len = 2
    seq_len = 4
    inp = rng.random((1, seq_len, inp_len))
    test1 = RNNSimple((seq_len, inp_len), out_len, Functions.tanh)
    res1, _ = test1.predict(inp)
    print(f"My res: {res1}, {res1.shape}")

    # w_ih = rng.random((1, inp_len, out_len))
    # w_hh = rng.random((1, out_len, out_len))
    # b_ih = rng.random((1, out_len))
    # b_hh = rng.random((1, out_len))
    #
    # w_ih = test1.weights[:inp_len].reshape((1, inp_len, out_len))
    # w_hh = test1.weights[inp_len:].reshape(1, out_len, out_len)
    # b_ih = test1.bias.reshape((1, out_len))
    # b_hh = np.zeros((1, out_len))


def test_backprop():
    inp_len = 5
    out_len = 2
    seq_len = 4
    inp = rng.random((5, seq_len, inp_len))
    out = rng.random((5, seq_len, out_len))
    test1 = RNNSimple((seq_len, inp_len), out_len, Functions.tanh)

    res1 = test1.forward(inp)
    errors = dmse(out, res1)
    test1.backward(errors, is_need_out=True)

def test_bacprop_2layers():
    pass
