from typing import Optional

import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse
from Node import Node, visualize_graph_with_networkx
import autograd_ops as autograd


class RNN(Layer):
    """
    Добавляет предыдущие состояния к сигналу перед Dense
    """
    # weights.shape = (input_size; output_size)
    weights: np.ndarray
    # bias.shape = (output_size)
    bias: np.ndarray
    input_signal: np.ndarray
    input_size: int
    seq_len: int
    output_size: int
    output_signal: list[Node]

    def __init__(self, input_size: tuple[int, int], output_size: int, activation_function: Functions):
        self.activation_function = activation_function
        self.input_size = input_size[1]
        self.input_shape = None
        self.seq_len = input_size[0]
        self.output_size = output_size
        self.weights = self._init_weights(size=(self.input_size + self.output_size, self.output_size))
        self.bias = self._init_biases(self.output_size)
        self.weights_node = Node(self.weights, name=f'weights')  # Узел для весов (один раз на слой)
        self.bias_node = Node(self.bias.reshape(1, -1), name=f'bias')  # Узел для bias (один раз на слой)
        self.params = {
            "weights": self.weights_node,
            "bias": self.bias_node,
        }

    def _init_weights(self, size: tuple):
        limit = np.sqrt(6. / (np.sum(size)))
        return np.array(rng.uniform(-limit, limit + 1e-5, size=size))

    def _init_biases(self, output_size):
        return np.array(rng.random(output_size) * 2 - 1)

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
        self.input_signal_nodes = []
        self.input_shape = len(signal)
        for t in range(self.seq_len):
            input_signal_node = signal[t]  # Создаём узел для input_signal
            past_node = self.nodes[-1] if len(self.nodes) != 0 else Node(np.zeros((batch, self.output_size)), name=f'Zerro')
            result_node = self._step(input_signal_node, past_node)

            self.input_signal_nodes.append(input_signal_node)  # Сохраняем узел для обратного прохода
            self.nodes.append(result_node)  # Сохраняем узел для обратного прохода
        return self.nodes

    def _step(self, input_signal_node: Node, past_node: Node) -> Node:
        """
        Выполняет один шаг RNN для входного и предыдущего состояния.
        """
        concat_node = autograd.concat([input_signal_node, past_node])
        linear_node = autograd.linear_act(self.activation_function,
                                          concat_node,
                                          self.weights_node,
                                          self.bias_node)
        return linear_node

    def update_params(self, learning_rate: Optional[float]):
        for param_name, param_node in self.params.items():
            param_node.value -= learning_rate * param_node.reset_grad()


def test_backprop():
    inp_len = 5
    out_len = 2
    seq_len = 4
    inp = get_test_inp()
    out = get_test_out()
    test1 = RNN((seq_len, inp_len), out_len, Functions.tanh)

    res1 = test1.predict(inp)
    # visualize_graph_with_networkx([test1.nodes[-1]])
    assert np.allclose(res1, get_expected_result())
    errors = dmse(res1, out)
    grad = test1.backward(errors, is_need_out=True, need_print=True)
    assert np.allclose(grad, get_expected_grad())
    print(test1.db)
    print(test1.dW)
    """
[[-2.61507166 -1.57878283]]
[[-1.41083347 -0.70988076]
 [-0.59815112 -0.6395795 ]
 [-0.69763604 -0.72547602]
 [-1.83847355 -0.63311828]
 [-1.35917772 -0.60912616]
 [-0.57442072 -0.26845245]
 [-0.91460504 -1.0046011 ]]"""

def test_2layers():
    inp_len = 5
    out_len1 = 3
    out_len2 = 2
    seq_len = 4
    inp = get_test_inp()
    # out = np.random.randint(0, 10, (inp_len, seq_len, out_len2)) / 5. - 1
    out = get_test_out()

    test1 = RNN((seq_len, inp_len), out_len1, Functions.tanh)
    test2 = RNN((seq_len, out_len1), out_len2, Functions.tanh)

    signal_nodes = [Node(inp[:, i], name=f'input[{i}]') for i in range(inp.shape[-2])]
    res1 = test1.predict(signal_nodes)
    res2 = test2.predict(res1)
    res = np.zeros_like(out)
    for t in range(seq_len):
        res[:, t] = res2[t].value
    errors = dmse(res, out)
    for t in reversed(range(seq_len - 1)):
        # Градиент ошибки для узла на текущем шаге
        res2[t].grad = errors[:, t]
    res2[-1].grad, res2[-1].ref_count = np.zeros_like(errors[:, -1]), 1
    res2[-1].backward(errors[:, -1], need_print=True)
    print(test1.bias_node.grad)
    print(test2.bias_node.grad)
    test1.update_params(learning_rate=1e-1)
    test2.update_params(learning_rate=1e-1)
    grad = np.zeros_like(inp)
    for t in range(seq_len):
        grad[:, t] = signal_nodes[t].reset_grad()
    print(grad)
    print(test1.bias_node.value)
    print(test2.bias_node.value)
    print(f"{test1.weights_node.value}")
    for _ in range(5):
        visualize_graph_with_networkx([test2.nodes[-1]])
    res1 = test1.predict(signal_nodes)
    res2 = test2.predict(res1)

    res1 = test1.predict(signal_nodes)
    res2 = test2.predict(res1)
    res = np.zeros_like(out)
    for t in range(seq_len):
        res[:, t] = res2[t].value
    errors = dmse(res, out)
    for t in reversed(range(seq_len - 1)):
        # Градиент ошибки для узла на текущем шаге
        res2[t].grad = errors[:, t]
    res2[-1].grad, res2[-1].ref_count = np.zeros_like(errors[:, -1]), 1
    res2[-1].backward(errors[:, -1], need_print=True)
    print(test1.bias_node.grad)
    print(test2.bias_node.grad)
    test1.update_params(learning_rate=1e-1)
    test2.update_params(learning_rate=1e-1)
    grad = np.zeros_like(inp)
    for t in range(seq_len):
        grad[:, t] = signal_nodes[t].reset_grad()
    print(grad)
    print(test1.bias_node.value)
    print(test2.bias_node.value)
    for _ in range(5):
        visualize_graph_with_networkx([test2.nodes[-1]])
    """[[[-0.0200466   0.05114068  0.11607556  0.15220417  0.03733221]
  [-0.03128068 -0.09390751  0.04965039  0.13608373 -0.05187273]
  [ 0.00529214  0.0194671   0.08700309  0.09806023  0.00846171]
  [-0.04481848 -0.02490972  0.04253059  0.11573787 -0.00170359]]

 [[-0.10905083  0.05970048 -0.12295093 -0.05572492  0.08097758]
  [-0.03644042 -0.00432579 -0.04196046 -0.01033099  0.01118185]
  [-0.00449316  0.06125032 -0.00957207 -0.03087868  0.04197394]
  [-0.02432664 -0.0227959   0.03672758  0.08393076 -0.00738095]]

 [[-0.2096918   0.21449342 -0.03968026  0.10674685  0.21553475]
  [ 0.06315429 -0.10877082  0.14966369  0.16176445 -0.09768713]
  [-0.01872001 -0.04207076 -0.00085244  0.03672275 -0.02093632]
  [-0.03981551  0.03494129 -0.04520569 -0.0258473   0.03818475]]

 [[-0.14211416  0.20688716  0.08430356  0.19123459  0.18327386]
  [-0.09354008 -0.04462833  0.11084353  0.26704352  0.00065067]
  [ 0.08455157 -0.05593052  0.15781447  0.12715926 -0.07084324]
  [ 0.01152732 -0.06120755  0.0438241   0.06675252 -0.0453803 ]]

 [[-0.08457609  0.05233176 -0.0926898  -0.04213338  0.06667925]
  [-0.01855579  0.03566336 -0.01071326 -0.00626844  0.03020188]
  [ 0.0334052   0.08134342  0.16167706  0.13763449  0.0370032 ]
  [ 0.10001808 -0.13746225  0.17572394  0.16400115 -0.13020363]]]
[[0.28605684 1.03374291 0.50190764]]
[[-1.33726735  6.2641866 ]]"""

def get_test_inp():
    batch_size = 5
    # out_len = 2
    seq_len = 4
    inp_len = 5
    inp = np.array([
        [[5, 8, 1, 9, 6],
         [1, 1, 3, 9, 7],
         [2, 1, 2, 8, 0],
         [3, 2, 0, 6, 0]],

        [[6, 6, 8, 1, 3],
         [3, 6, 1, 1, 4],
         [0, 8, 8, 9, 4],
         [4, 8, 3, 2, 0]],

        [[9, 3, 8, 4, 4],
         [4, 4, 8, 2, 7],
         [5, 1, 4, 8, 8],
         [7, 1, 7, 2, 4]],

        [[4, 3, 3, 6, 1],
         [1, 6, 1, 5, 3],
         [9, 3, 1, 5, 3],
         [3, 5, 6, 7, 2]],

        [[4, 9, 6, 6, 6],
         [1, 8, 8, 9, 8],
         [3, 2, 1, 3, 1],
         [9, 1, 4, 6, 4]]
    ]) / 10.
    return inp


def get_test_out():
    batch_size = 5
    seq_len = 4
    # inp_len = 5
    out_len = 2
    outp = np.array([
        [[2, 5],
         [0, 5],
         [2, 4],
         [6, 9]],

        [[5, 5],
         [8, 8],
         [8, 5],
         [6, 9]],

        [[8, 6],
         [0, 5],
         [6, 6],
         [8, 4]],

        [[3, 8],
         [1, 9],
         [2, 1],
         [5, 6]],

        [[6, 7],
         [9, 8],
         [6, 0],
         [2, 6]]
    ])/5 - 1
    return outp

def get_expected_result():
    return np.array([
        [[ 0.2603783,   0.63554271],
         [ 0.15190845,  0.90184053],
         [-0.00493864,  0.90953664],
         [-0.268465,    0.88386899]],

        [[ 0.36694794,  0.18386528],
         [-0.03050504,  0.69215378],
         [-0.15251228,  0.69310605],
         [-0.44532675,  0.56560038]],

        [[ 0.69559701,  0.3275082 ],
         [ 0.53632462,  0.68363741],
         [ 0.52510433,  0.88982901],
         [ 0.45861279,  0.79141646]],

        [[ 0.38161309,  0.66892566],
         [-0.22314013,  0.8544933 ],
         [-0.03792111,  0.73733143],
         [-0.01516243,  0.73029642]],

        [[ 0.23374859,  0.37271413],
         [ 0.22092953,  0.67110957],
         [-0.10489812,  0.84602062],
         [ 0.29431428,  0.75557195]]
    ])

def get_expected_grad():
    return np.array([
        [[-0.5192052,   0.27029487, -0.34300748, -0.34227874,  0.0898885],
         [-0.39661869,  0.25814531, -0.24070272, -0.31067455,  0.0743485],
         [-0.06087936,  0.09280292, -0.01500438, -0.09833513,  0.01726138],
         [ 0.12794232, -0.06626828,  0.08466308,  0.08402258, -0.02211314]],

        [[ 0.17251841,  0.15720784,  0.21589752, -0.1215336,  -0.00269759],
         [ 0.39903296, -0.09881262,  0.30855996,  0.15931917, -0.05710302],
         [ 0.39167113, -0.09689436,  0.30290658,  0.15628915, -0.05603904],
         [ 0.129434,   -0.11170729,  0.06721997,  0.12754288, -0.02728387]],

        [[-0.2208521,   0.01400698, -0.1875646,  -0.04943156,  0.02712997],
         [-0.33790847,  0.26544957, -0.1862912,  -0.3080369,   0.06834937],
         [-0.06557693,  0.0835403,  -0.02293877, -0.09028099,  0.01678687],
         [ 0.07893533,  0.04924568,  0.08942333, -0.03400259, -0.00372935]],

        [[-0.45588095,  0.1589698,  -0.33350533, -0.2259034,   0.07030656],
         [-0.37976985,  0.20915621, -0.24616629, -0.26126349,  0.06700791],
         [-0.09856626,  0.24317225,  0.014048,   -0.24770697,  0.03816734],
         [ 0.0355564,   0.04241109,  0.04862734, -0.03458216,  0.00054506]],

        [[ 0.01934462,  0.00235672,  0.01790761,  0.00091669, -0.00198217],
         [ 0.04288326, -0.01765545,  0.03025701,  0.0238231,  -0.00691067],
         [-0.04822334,  0.13237577,  0.0124038,  -0.13395645,  0.02014764],
         [-0.20604046,  0.17396651, -0.10859548, -0.199358,    0.04300796]]
    ])