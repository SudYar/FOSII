from typing import Optional

import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse, initialize_weights
from Node import Node, visualize_graph_with_networkx
import autograd_ops as autograd


class GRU(Layer):
    """
    Добавляет предыдущие состояния к сигналу перед Dense
    """
    # weights.shape = (input_size; output_size)
    update_gate_weights: Node
    reset_gate_weights: Node
    candidate_weights: Node
    # bias.shape = (1, output_size)
    bias_z: Node
    bias_r: Node
    bias_h: Node

    input_size: int
    seq_len: int
    sigm_f = Functions.sigmoid
    tanh_f = Functions.tanh
    output_size: int
    output_signal: list[Node]

    def __init__(self, input_size: tuple[int, int], output_size: int, activation_function: Functions):
        self.activation_function = activation_function
        self.input_size = input_size[1]
        self.seq_len = input_size[0]
        self.output_size = output_size
        self.input_size_with_h = self.input_size + output_size

        def init_w():
            return initialize_weights(in_size=self.input_size_with_h,
                                      size=(self.input_size_with_h, self.output_size),
                                      out_size=self.output_size, method='ksa')

        self.update_gate_weights = Node(init_w(), name='upd_w')
        self.reset_gate_weights = Node(init_w(), name='res_w')
        self.candidate_weights = Node(init_w(), name='can_w')
        self.bias_z = Node(self._init_biases(), name='upd_b')
        self.bias_r = Node(self._init_biases(), name='res_b')
        self.bias_h = Node(self._init_biases(), name='can_b')
        self.params = {
            "upd_w": self.update_gate_weights,
            "res_w": self.reset_gate_weights,
            "can_w": self.candidate_weights,
            "upd_b": self.bias_z,
            "res_b": self.bias_r,
            "can_b": self.bias_h
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
        for t in range(self.seq_len):
            input_signal_node = signal[t]  # Создаём узел для input_signal
            past_node = self.nodes[-1] if len(self.nodes) != 0 else Node(np.zeros((batch, self.output_size)),
                                                                         name=f'Zerro')
            result_node = self._step(input_signal_node, past_node)

            self.nodes.append(result_node)  # Сохраняем узел для обратного прохода
        return self.nodes

    def _step(self, input_signal_node: Node, past_node: Node) -> Node:
        """
        Выполняет один шаг RNN для входного и предыдущего состояния.
        """
        concat_node = autograd.concat([input_signal_node, past_node])

        z_gate = autograd.linear_act(self.sigm_f, concat_node, self.update_gate_weights, self.bias_z, name='z_gate(act)')

        r_gate = autograd.linear_act(self.sigm_f, concat_node, self.reset_gate_weights, self.bias_r, name='r_gate(act)')

        reset_info = autograd.multiply(past_node, r_gate)
        reset_inp = autograd.concat([input_signal_node, reset_info])
        # reset_inp = autograd.multiply_with_slice(concat_node, r_gate, (self.input_size, None))
        h_hat_act = autograd.linear_act(self.activation_function, reset_inp, self.candidate_weights, self.bias_h, name='h_hat(act)')

        update_gate = autograd.minus(np.ones_like(z_gate.value), z_gate)
        updated_past = autograd.multiply(update_gate, past_node, name='i_old')
        updated_new = autograd.multiply(z_gate, h_hat_act, name='i_new')
        return autograd.add(updated_past, updated_new)

    def update_params(self, learning_rate: Optional[float]):
        for param_name, param_node in self.params.items():
            param_node.value -= learning_rate * param_node.reset_grad()


def test_forward():
    inp_len = 5
    out_len = 2
    seq_len = 4
    inp = rng.random((1, seq_len, inp_len))
    test1 = GRU((seq_len, inp_len), out_len, Functions.tanh)
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
    need_visual = False
    inp_len = 5
    out_len = 2
    seq_len = 4
    # inp = rng.random((5, seq_len, inp_len))
    # out = rng.random((5, seq_len, out_len))
    inp = get_test_inp()
    out = get_test_out()
    test1 = GRU((seq_len, inp_len), out_len, Functions.tanh)
    signal_nodes = [Node(inp[:, i], name=f'input[{i}]') for i in range(inp.shape[-2])]
    res1 = test1.predict(signal_nodes)
    res = np.array([node.value for node in res1]).transpose((1, 0, 2))
    if need_visual:
        for _ in range(2):
            visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    errors = dmse(res, out)
    print(f"{errors=}")
    for t in reversed(range(seq_len - 1)):
        # Градиент ошибки для узла на текущем шаге
        res1[t].grad = errors[:, t]
    res1[-1].grad, res1[-1].ref_count = np.zeros_like(errors[:, -1]), 1
    res1[-1].backward(errors[:, -1], need_print=True)
    print(f"{test1.bias_z.grad=}")
    test1.update_params(learning_rate=1e-1)
    if need_visual:
        for _ in range(2):
            visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    res1 = test1.predict(signal_nodes)
    res = np.array([node.value for node in res1]).transpose((1, 0, 2))
    if need_visual:
        for _ in range(2):
            visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    errors = dmse(res, out)
    print(f"{errors=}")
    # res1 = test1.forward(inp)
    # errors = dmse(out, res1)
    # test1.backward(errors, is_need_out=True)
    """test1.bias_z.grad=array([[0.25686655, 1.28804842]])"""

from RNN import get_test_inp, get_test_out
def test_2layers():
    inp_len = 5
    out_len1 = 3
    out_len2 = 2
    seq_len = 4
    inp = get_test_inp()
    # out = np.random.randint(0, 10, (inp_len, seq_len, out_len2)) / 5. - 1
    outp = get_test_out()

    test1 = GRU((seq_len, inp_len), out_len1, Functions.tanh)
    test2 = GRU((seq_len, out_len1), out_len2, Functions.tanh)
    print(f"{test2.candidate_weights.value=}")

    signal_nodes = [Node(inp[:, i], name=f'input[{i}]') for i in range(inp.shape[-2])]
    res1 = test1.predict(signal_nodes)
    res2 = test2.predict(res1)
    res = np.zeros_like(outp)
    for t in range(seq_len):
        res[:, t] = res2[t].value
    # visualize_graph_with_networkx([res1[-1]], scale=5, node_size=500, font_size=6)
    print(f"{np.array([node.value for node in res2]).transpose(1,0,2)=}")
    errors = dmse(res, outp)
    print(f"{errors=},\n{mse(res, outp)=}")
    for t in reversed(range(seq_len - 1)):
        # Градиент ошибки для узла на текущем шаге
        res2[t].grad = errors[:, t]
    res2[-1].grad, res2[-1].ref_count = np.zeros_like(errors[:, -1]), 1
    res2[-1].backward(errors[:, -1])#, need_print=True)
    # visualize_graph_with_networkx([res1[-1]], scale=5, node_size=1000, font_size=8)
    print(f"{test1.bias_z.grad=}")
    print(f"{test1.bias_h.grad=}")
    test1.update_params(learning_rate=1e-1)
    test2.update_params(learning_rate=1e-1)
    grad = np.zeros_like(inp)
    for t in range(seq_len):
        grad[:, t] = signal_nodes[t].reset_grad()
    print(f"{grad=}")
    print(f"{test1.bias_z.value=}")
    print(f"{test1.bias_h.value=}")
    print(f"{test1.candidate_weights.value=}")
    # for _ in range(5):
    #     visualize_graph_with_networkx([test2.nodes[-1]])
    res1 = test1.predict(signal_nodes)
    res2 = test2.predict(res1)

    res1 = test1.predict(signal_nodes)
    res2 = test2.predict(res1)
    res = np.zeros_like(outp)
    for t in range(seq_len):
        res[:, t] = res2[t].value
    errors2 = dmse(res, outp)
    print(f"{mse(res, outp)=}")
    for t in reversed(range(seq_len - 1)):
        # Градиент ошибки для узла на текущем шаге
        res2[t].grad = errors2[:, t]
    res2[-1].grad, res2[-1].ref_count = np.zeros_like(errors2[:, -1]), 1
    res2[-1].backward(errors2[:, -1])#, need_print=True)
    print(f"{test1.bias_z.grad=}")
    print(f"{test1.bias_h.grad=}")
    test1.update_params(learning_rate=1e-1)
    test2.update_params(learning_rate=1e-1)
    grad = np.zeros_like(inp)
    for t in range(seq_len):
        grad[:, t] = signal_nodes[t].reset_grad()
    print(f"{grad=}")

    for param_name, param_node in test1.params.items():
        if '_b' in param_name:
            print(f"{param_name}: {param_node.value}")

    # for _ in range(5):
        # visualize_graph_with_networkx([test2.nodes[-1]])


    """test1.bias_z.value=array([[0.25809999, 0.99760529, 0.79380637]])
test1.bias_h.value=array([[-0.1574161 , -0.14328938, -0.32858012]])
upd_b: [[0.25809999 0.99760529 0.79380637]]
res_b: [[-0.81381375 -0.51929722  0.08616757]]
can_b: [[-0.1574161  -0.14328938 -0.32858012]]"""