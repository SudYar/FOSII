from typing import Optional

import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse, initialize_weights


class LSTMLayer(Layer):
    """
    Реализация LSTM слоя
    """
    # weights.shape = (input_size+out_size; output_size)
    input_gate_weights: np.ndarray
    output_gate_weights: np.ndarray
    forget_weights: np.ndarray
    candidate_weights: np.ndarray
    # bias.shape = (output_size)
    input_gate_bias: np.ndarray
    output_gate_bias: np.ndarray
    forget_bias: np.ndarray
    candidate_bias: np.ndarray
    # delta
    dW_in: np.ndarray
    dW_out: np.ndarray
    dW_forgot: np.ndarray
    dW_cell: np.ndarray
    db_in: np.ndarray
    db_out: np.ndarray
    db_forgot: np.ndarray
    db_cell: np.ndarray
    input_signal: np.ndarray
    input_size: int
    seq_len: int
    sigm_f = Functions.sigmoid
    tanh_f = Functions.tanh
    signal_h: np.ndarray
    forget_gate: np.ndarray
    in_gate: np.ndarray
    out_gate: np.ndarray
    C_tilda: np.ndarray
    cell: np.ndarray
    output_size: int
    output_signal: np.ndarray

    def __init__(self, input_size: tuple[int, int], output_size: int, activation_function: Functions=Functions.tanh):
        self.activation_function = activation_function
        self.input_size = input_size[1]
        self.seq_len = input_size[0]
        self.output_size = output_size
        self.input_size_with_h = self.input_size + output_size
        self.input_gate_weights = initialize_weights(in_size=self.input_size_with_h,
                                                     size=(self.input_size_with_h, self.output_size),
                                                     out_size=self.output_size, method='ksa')
        self.output_gate_weights = initialize_weights(in_size=self.input_size_with_h,
                                                      size=(self.input_size_with_h, self.output_size),
                                                      out_size=self.output_size, method='ksa')
        self.forget_weights = initialize_weights(in_size=self.input_size_with_h,
                                                 size=(self.input_size_with_h, self.output_size),
                                                 out_size=self.output_size, method='ksa')
        self.candidate_weights = initialize_weights(in_size=self.input_size_with_h,
                                                    size=(self.input_size_with_h, self.output_size),
                                                    out_size=self.output_size, method='ksa')

        self.input_gate_bias = self._init_biases(self.output_size)
        self.output_gate_bias = self._init_biases(self.output_size)
        self.forget_bias = self._init_biases(self.output_size)#np.ones(self.output_size)
        self.candidate_bias = self._init_biases(self.output_size)

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
        self.cell = np.zeros_like(result)
        self.in_gate, self.out_gate, self.forget_gate, self.C_tilda = \
            (np.zeros_like(result) for _ in range(4))
        self.input_signal = np.concatenate((signal, result), axis=-1)
        for t in range(self.seq_len):
            if t > 0:
                self.input_signal[:, t, self.input_size:] = result[:, t - 1]
            x_h_t = self.input_signal[:, t] # чтобы короче дальше писать
            self.forget_gate[:, t] = self.sigm_f.calc(
                np.matmul(x_h_t, self.forget_weights) + self.forget_bias
            )
            self.out_gate[:, t] = self.sigm_f.calc(
                np.matmul(x_h_t, self.output_gate_weights) + self.output_gate_bias
            )
            self.in_gate[:, t] = self.sigm_f.calc(
                np.matmul(x_h_t, self.input_gate_weights) + self.input_gate_bias
            )
            self.C_tilda[:, t] = self.tanh_f.calc(
                np.matmul(x_h_t, self.candidate_weights) + self.candidate_bias
            )

            if t > 0:
                self.cell[:, t] = self.forget_gate[:, t] * self.cell[:, t-1]
            self.cell[:, t] += self.in_gate[:, t] * self.C_tilda[:, t]

            result[:, t] = self.out_gate[:, t] * self.activation_function.calc(self.cell[:, t]) # по дефолту tanh. Не знаю, для чего может пригодиться
        return result

    def backward(self, error_signal: np.ndarray, is_need_out=True) -> np.ndarray | None:
        """

        :param error_signal: переданный дельта с предыдущих слоев (размерности (batch, sequence_len, out))
        :param is_need_out: Необходимость проталкивать delta дальше. Если это первый слой, то нет необходимости
        :return:
        """
        result = np.zeros_like(self.input_signal)
        self.dW_in, self.dW_out, self.dW_forgot, self.dW_cell = \
            (np.zeros_like(self.input_gate_weights) for _ in range(4))
        self.db_in, self.db_out, self.db_forgot, self.db_cell = \
            (np.zeros_like(self.input_gate_bias) for _ in range(4))
        for t in reversed(range(self.seq_len)):
            # f (batch, out) * ((batch, out) + (batch, out) or 0)
            delta_t = error_signal[:, t] + (delta_t_plus_1 if t + 1 < self.seq_len else 0)
            delta_out_gate = delta_t * self.activation_function.calc(self.cell[:, t]) \
                                     * self.sigm_f.derivative(self.out_gate[:, t])
            delta_cell = delta_t * self.out_gate[:, t]\
                                 * self.activation_function.derivative(self.output_signal[:, t])\
                        + (delta_cell_plus_1 if t + 1 < self.seq_len else 0)

            delta_in_gate = delta_cell * self.C_tilda[:, t] * self.sigm_f.derivative(self.in_gate[:, t])
            delta_forgot_gate = delta_cell * self.cell[:, t-1] * self.sigm_f.derivative(self.forget_gate[:, t])
            delta_c_tilda = delta_cell * self.in_gate[:, t] * self.tanh_f.derivative(self.C_tilda[:, t])

            x_h_t_T = self.input_signal[:, t].T
            self.dW_in += np.matmul(x_h_t_T, delta_in_gate)
            self.dW_out += np.matmul(x_h_t_T, delta_out_gate)
            self.dW_cell += np.matmul(x_h_t_T, delta_c_tilda)
            self.dW_forgot += np.matmul(x_h_t_T, delta_forgot_gate)

            self.db_in += np.sum(delta_in_gate, axis=0)
            self.db_out += np.sum(delta_out_gate, axis=0)
            self.db_cell += np.sum(delta_c_tilda, axis=0)
            self.db_forgot += np.sum(delta_forgot_gate, axis=0)

            result[:, t] = (+ np.matmul(delta_in_gate, self.input_gate_weights.T)
                            + np.matmul(delta_out_gate, self.output_gate_weights.T)
                            + np.matmul(delta_c_tilda, self.candidate_weights.T)
                            + np.matmul(delta_forgot_gate, self.forget_weights.T)
                            )
            # (batch, in+out).T x (batch, out) -> (in+out, out)
            # self.dW += np.matmul(self.input_signal[:, t].T, delta_t)
            # self.db += np.sum(delta_t, axis=0)
            # delta, чтобы передать для t-1;    # веса для скрытого слоя размерности V(out, out)
            delta_t_plus_1 = result[:, t, self.input_size:] if t != 0 else 0  # условие чтобы не считать при t = 0
            delta_cell_plus_1 = (delta_cell * self.forget_gate[:, t]) if t != 0 else 0

        return result[:, :, :self.input_size] if is_need_out else None

    def update_params(self, learning_rate: Optional[float]):
        self.input_gate_weights -= learning_rate * self.dW_in
        self.output_gate_weights -= learning_rate * self.dW_out
        self.forget_weights -= learning_rate * self.dW_forgot
        self.candidate_weights -= learning_rate * self.dW_cell
        self.input_gate_bias -= learning_rate * self.db_in
        self.output_gate_bias -= learning_rate * self.db_out
        self.forget_bias -= learning_rate * self.db_forgot
        self.candidate_bias -= learning_rate * self.db_cell


def test_init_weigh():
    inp_size, seq_len, outp_size = 5, 6, 3
    test = LSTMLayer((seq_len, inp_size), outp_size, Functions.none)
    print(f"{test.input_gate_weights},\n\n {test.output_gate_weights},\n\n {test.candidate_weights}")

def test_forward():
    inp_len = 5
    out_len = 2
    seq_len = 4
    inp = rng.random((1, seq_len, inp_len))
    test1 = LSTMLayer((seq_len, inp_len), out_len, Functions.tanh)
    res1 = test1.predict(inp)
    print(f"My res: {res1}, {res1.shape}")

def test_backprop():
    inp_len = 6
    out_len = 2
    seq_len = 4
    inp = rng.random((5, seq_len, inp_len))
    out = rng.random((5, seq_len, out_len))
    test1 = LSTMLayer((seq_len, inp_len), out_len, Functions.tanh)

    res1 = test1.forward(inp)
    errors = dmse(out, res1)
    test1.backward(errors, is_need_out=True)