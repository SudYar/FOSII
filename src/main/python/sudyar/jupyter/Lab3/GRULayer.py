from typing import Optional

import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse, initialize_weights


class GRULayer(Layer):
    """
    Добавляет предыдущие состояния к сигналу перед Dense
    """
    # weights.shape = (input_size; output_size)
    update_gate_weights: np.ndarray
    reset_gate_weights: np.ndarray
    candidate_weights: np.ndarray
    # bias.shape = (output_size)
    bias_z: np.ndarray
    bias_r: np.ndarray
    bias_h: np.ndarray

    # delta
    dW_z: np.ndarray
    dW_r: np.ndarray
    dW_h: np.ndarray
    db_z: np.ndarray
    db_r: np.ndarray
    db_h: np.ndarray

    input_signal: np.ndarray
    input_signal_r: np.ndarray
    z: np.ndarray
    r: np.ndarray
    h: np.ndarray
    h_hat: np.ndarray

    input_size: int
    seq_len: int
    sigm_f = Functions.sigmoid
    tanh_f = Functions.tanh
    output_size: int
    output_signal: np.ndarray

    def __init__(self, input_size: tuple[int, int], output_size: int, activation_function: Functions):
        self.activation_function = activation_function
        self.input_size = input_size[1]
        self.seq_len = input_size[0]
        self.output_size = output_size
        self.input_size_with_h = self.input_size + output_size
        self.update_gate_weights = initialize_weights(in_size=self.input_size_with_h,
                                                      size=(self.input_size_with_h, self.output_size),
                                                      out_size=self.output_size, method='ksa')
        self.reset_gate_weights = initialize_weights(in_size=self.input_size_with_h,
                                                     size=(self.input_size_with_h, self.output_size),
                                                     out_size=self.output_size, method='ksa')
        self.candidate_weights = initialize_weights(in_size=self.input_size_with_h,
                                                    size=(self.input_size_with_h, self.output_size),
                                                    out_size=self.output_size, method='ksa')

        self.bias_z = self._init_biases(self.output_size)
        self.bias_r = self._init_biases(self.output_size)
        self.bias_h = self._init_biases(self.output_size)  # np.ones(self.output_size)

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
        # Инициализация массивов для хранения временных состояний
        self.h, self.z, self.r, self.h_hat = \
            (np.zeros_like(result) for _ in range(4))

        self.input_signal = np.concatenate((signal, result), axis=-1)
        self.input_signal_r = self.input_signal.copy()
        for t in range(self.seq_len):
            if t > 0:
                self.input_signal[:, t, self.input_size:] = result[:, t - 1]
            x_h_t = self.input_signal[:, t]  # чтобы проще дальше считать

            self.z[:, t] = self.sigm_f.calc(np.matmul(x_h_t, self.update_gate_weights) + self.bias_z)
            self.r[:, t] = self.sigm_f.calc(np.matmul(x_h_t, self.reset_gate_weights) + self.bias_r)

            self.input_signal_r[:, t, self.input_size:] = self.r[:, t] * x_h_t[:, self.input_size:]
            self.h_hat[:, t] = self.activation_function.calc(
                np.matmul(self.input_signal_r[:, t], self.candidate_weights) + self.bias_h
            )

            result[:, t] = (1 - self.z[:, t]) * result[:, t - 1] + self.z[:, t] * self.h_hat[:, t]
        return result

    def backward(self, error_signal: np.ndarray, is_need_out=True) -> np.ndarray | None:
        """

        :param error_signal: переданный дельта с предыдущих слоев (размерности (batch, sequence_len, out))
        :param is_need_out: Необходимость проталкивать delta дальше. Если это первый слой, то нет необходимости
        :return:
        """
        result = np.zeros_like(self.input_signal[:, :, :self.input_size])
        self.dW_z, self.dW_r, self.dW_h = \
            (np.zeros_like(self.update_gate_weights) for _ in range(3))
        self.db_z, self.db_r, self.db_h = \
            (np.zeros_like(self.bias_z) for _ in range(3))
        for t in reversed(range(self.seq_len)):
            # f (batch, out) * ((batch, out) + (batch, out) or 0)
            sum_error = error_signal[:, t] + (delta_t_plus_1 if t + 1 < self.seq_len else 0)
            delta_z = sum_error * (self.h_hat[:, t] - self.input_signal[:, t, self.input_size:]) \
                                * self.sigm_f.derivative(self.z[:, t])
            delta_h_hat = sum_error * self.z[:, t] \
                                    * self.activation_function.derivative(self.h_hat[:, t])
            delta_r = delta_h_hat * np.matmul(self.input_signal[:, t, self.input_size:],
                                              self.candidate_weights[self.input_size:].T) \
                                  * self.sigm_f.derivative(self.r[:, t])
            # (batch, in+out).T x (batch, out) -> (in+out, out)
            self.dW_z += np.matmul(self.input_signal[:, t].T, delta_z)
            self.dW_r += np.matmul(self.input_signal[:, t].T, delta_r)
            self.dW_h += np.matmul(self.input_signal_r[:, t].T, delta_h_hat)

            self.db_z += np.sum(delta_z, axis=0)
            self.db_r += np.sum(delta_r, axis=0)
            self.db_h += np.sum(delta_h_hat, axis=0)
            # delta, чтобы передать для t-1;    # веса для скрытого слоя размерности V(out, out)
            delta_t_plus_1 = (+ sum_error * (1 - self.z[:, t])
                              + delta_h_hat *  np.matmul(self.r[:, t], self.reset_gate_weights[self.input_size:].T)
                              + np.matmul(delta_r, self.update_gate_weights[self.input_size:].T)
                              + np.matmul(delta_z, self.update_gate_weights[self.input_size:].T)) if t != 0 else 0  # условие чтобы не считать при t = 0
            # # [error(batch, out)x w(in, out).T  ] -> (batch, in)
            if is_need_out:
                result[:, t] = (
                        + np.matmul(delta_r, self.update_gate_weights[:self.input_size].T)
                        + np.matmul(delta_z, self.update_gate_weights[:self.input_size].T)
                        + np.matmul(delta_h_hat, self.candidate_weights[:self.input_size].T)
                )
        return result if is_need_out else None

    def update_params(self, learning_rate: Optional[float]):
        self.update_gate_weights -= learning_rate * self.dW_z
        self.reset_gate_weights -= learning_rate * self.dW_r
        self.candidate_weights -= learning_rate * self.dW_h
        self.bias_z -= learning_rate * self.db_z
        self.bias_r -= learning_rate * self.db_r
        self.bias_h -= learning_rate * self.db_h


def test_forward():
    inp_len = 5
    out_len = 2
    seq_len = 4
    inp = rng.random((1, seq_len, inp_len))
    test1 = GRULayer((seq_len, inp_len), out_len, Functions.tanh)
    res1 = test1.predict(inp)
    print(f"My res: {res1}, {res1.shape}")

    # w_ih = rng.random((1, inp_len, out_len))
    # w_hh = rng.random((1, out_len, out_len))
    # b_ih = rng.random((1, out_len))
    # b_hh = rng.random((1, out_len))

    # w_ih = test1.weights[:inp_len].reshape((1, inp_len, out_len))
    # w_hh = test1.weights[inp_len:].reshape(1, out_len, out_len)
    # b_ih = test1.bias.reshape((1, out_len))
    # b_hh = np.zeros((1, out_len))


def test_backprop():
    inp_len = 6
    out_len = 2
    seq_len = 4
    inp = rng.random((5, seq_len, inp_len))
    out = rng.random((5, seq_len, out_len))
    test1 = GRULayer((seq_len, inp_len), out_len, Functions.tanh)

    res1 = test1.forward(inp)
    errors = dmse(out, res1)
    test1.backward(errors, is_need_out=True)


def test_bacprop_2layers():
    pass
