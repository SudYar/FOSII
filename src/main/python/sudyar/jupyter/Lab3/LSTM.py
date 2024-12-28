from typing import Optional

import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse, initialize_weights
from Node import Node
import autograd_ops as autograd
from src.main.python.sudyar.jupyter.Lab3.RNN import get_test_inp, get_test_out


class LSTM(Layer):
    """
    Реализация LSTM слоя
    """
    # weights.shape = (input_size+out_size; output_size)
    input_gate_weights: Node
    output_gate_weights: Node
    forget_weights: Node
    candidate_weights: Node
    # bias.shape = (output_size)
    input_gate_bias: Node
    output_gate_bias: Node
    forget_bias: Node
    candidate_bias: Node

    input_size: int
    seq_len: int
    sigm_f = Functions.sigmoid
    tanh_f = Functions.tanh
    cell: Node
    output_size: int
    output_signal: list[Node]

    def __init__(self, input_size: tuple[int, int], output_size: int, activation_function: Functions=Functions.tanh):
        self.activation_function = activation_function
        self.input_size = input_size[1]
        self.seq_len = input_size[0]
        self.output_size = output_size
        self.input_size_with_h = self.input_size + output_size

        def init_w():
            return initialize_weights(in_size=self.input_size_with_h,
                                      size=(self.input_size_with_h, self.output_size),
                                      out_size=self.output_size, method='ksa')

        self.input_gate_weights = Node(init_w(), name='in_gate_w')
        self.output_gate_weights = Node(init_w(), name='out_gate_w')
        self.forget_weights = Node(init_w(), name='forget_w')
        self.candidate_weights = Node(init_w(), name='can_w')

        self.input_gate_bias = Node(self._init_biases(), name='in_gate_b')
        self.output_gate_bias = Node(self._init_biases(), name='out_gate_b')
        self.forget_bias = Node(self._init_biases(), name='forget_b')
        self.candidate_bias = Node(self._init_biases(), name='can_b')

        self.params = {
            "in_gate_w": self.input_gate_weights,
            "out_gate_w": self.output_gate_weights,
            "forget_w": self.forget_weights,
            "can_w": self.candidate_weights,
            "in_gate_b": self.input_gate_bias,
            "out_gate_b": self.output_gate_bias,
            "forget_b": self.forget_bias,
            "can_b": self.candidate_bias
        }

    def _init_biases(self):
        return np.array(rng.random(self.output_size) * 2 - 1).reshape(1, -1)

    def forward(self, signal: list[Node]) -> list[Node]:
        """

        :param signal: np.ndarray - Входящий сигнал, размерности (batch, sequence_length, output_size)
        :return: np.ndarray - выходящий сигнал размерности (batch, sequence_length, output_size
        """

        #         in_sig(batch, in), w (in, out), + b (out ,)
        self.output_signal = self.predict(signal)
        return self.output_signal

    def predict(self, signal: list[Node]) -> list[Node]:
        batch, inp_len = signal[0].value.shape
        self.nodes = []  # Храним узлы для backward
        self.input_shape = len(signal)
        self.cell = Node(np.zeros((batch, self.output_size)), name='cell')
        for t in range(self.seq_len):
            input_signal_node = signal[t]  # Создаём узел для input_signal
            past_node = self.nodes[-1] if len(self.nodes) != 0 else Node(np.zeros((batch, self.output_size)),
                                                                         name=f'Zerro')
            result_node, self.cell = self._step(input_signal_node, past_node, self.cell)
            self.nodes.append(result_node)

        return self.nodes

    def _step(self, input_signal_node: Node, past_node: Node, C_t_old: Node) -> tuple[Node, Node]:
        """
        Выполняет один шаг RNN для входного и предыдущего состояния.
        """
        concat_node = autograd.concat([input_signal_node, past_node], name='x_h_t')
        forget_gate = autograd.linear_act(self.sigm_f, concat_node, self.forget_weights, self.forget_bias, name='f_t')
        in_gate = autograd.linear_act(self.sigm_f, concat_node, self.input_gate_weights, self.input_gate_bias, name='i_t')
        C_tilda = autograd.linear_act(self.tanh_f, concat_node, self.candidate_weights, self.candidate_bias, name='C^hat_t')
        new_cell = autograd.multiply(in_gate, C_tilda)
        out_gate = autograd.linear_act(self.sigm_f, concat_node, self.output_gate_weights, self.output_gate_bias, name='o_t')

        forget_cell = autograd.multiply(C_t_old, forget_gate)
        cell = autograd.add(forget_cell, new_cell)

        result = autograd.act(self.activation_function, cell)

        return autograd.multiply(out_gate, result), cell

    def update_params(self, learning_rate: Optional[float]):
        for param_name, param_node in self.params.items():
            param_node.value -= learning_rate * param_node.reset_grad()


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
    need_visual = False
    inp_len = 5
    out_len = 2
    seq_len = 4
    # inp = rng.random((5, seq_len, inp_len))
    # out = rng.random((5, seq_len, out_len))
    inp = get_test_inp()
    out = get_test_out()
    test1 = LSTMLayer((seq_len, inp_len), out_len, Functions.tanh)
    signal_nodes = [Node(inp[:, i], name=f'input[{i}]') for i in range(inp.shape[-2])]
    res1 = test1.predict(signal_nodes)
    res = np.array([node.value for node in res1]).transpose((1, 0, 2))
    # if need_visual:
    #     for _ in range(2):
    #         visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    errors = dmse(res, out)
    print(f"{errors=}")
    for t in reversed(range(seq_len - 1)):
        # Градиент ошибки для узла на текущем шаге
        res1[t].grad = errors[:, t]
    res1[-1].grad, res1[-1].ref_count = np.zeros_like(errors[:, -1]), 1
    res1[-1].backward(errors[:, -1], need_print=True)
    print(f"{test1.forget_bias.grad=}")
    test1.update_params(learning_rate=1e-1)
    # if need_visual:
    #     for _ in range(2):
    #         visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    res1 = test1.predict(signal_nodes)
    res = np.array([node.value for node in res1]).transpose((1, 0, 2))
    # if need_visual:
        # for _ in range(2):
        #     visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    errors = dmse(res, out)
    print(f"{errors=}")
    # res1 = test1.forward(inp)
    # errors = dmse(out, res1)
    # test1.backward(errors, is_need_out=True)