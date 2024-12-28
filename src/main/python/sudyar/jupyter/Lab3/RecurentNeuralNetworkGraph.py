import math

from RecurrentLayerTypes import RecurrentLayerTypes, create_layer
from Layer import Layer
from Node import Node

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Functions import mse, dmse, Functions, rng


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])


class RecurentNeuralNetworkGraph:
    layers: list[Layer]
    learning_rate: float

    def __init__(self, architecture: list[int], layers_type: RecurrentLayerTypes, sequence_len: int, learning_rate: float):
        """

        Пример architecture= [6, 5, 3] означает что будет два слоя: 1 (6, 5); 2 (5, 3)

        :param architecture: Количество нейронов на каждом слое
        :param sequence_len: Длина временной последовательности
        :param learning_rate:
        """
        if len(architecture) < 2:
            raise Exception("Model is empty")

        self.layers = self._build_layers(architecture, sequence_len, layers_type)
        self.learning_rate = learning_rate

    def _build_layers(self, arch: list, sequence_len: int, layer_type: RecurrentLayerTypes):
        result = []
        for i in range(len(arch)-1):
            fun = Functions.tanh #if i < len(arch)-2 else Functions.none
            result.append(create_layer(layer_type,
                                       input_size=(sequence_len, arch[i]), output_size=arch[i+1],
                                       activation_function=fun))
        return result
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        self.forward(input_data)
        res_nodes = self.output_layer.nodes
        res = np.array([node.value for node in res_nodes]).transpose(1,0,2)
        return res

    def forward(self, input_data: np.ndarray):
        """
        Прямой проход
        :param input_data: входящие данные формата (l, seq_len, in)
        :return: result формата (l, seq_len, out)
        """
        signal = input_data
        signal_nodes = [Node(signal[:, i], name=f'input[{i}]') for i in range(signal.shape[-2])]
        for layer in self.layers:
            signal_nodes = layer.predict(signal_nodes)

    def backward(self, y_test: np.array) -> None:
        # out_s (l ,out), y_test (batch, out) -> (l, out)

        batch, seq_len, out_len = y_test.shape
        output_nodes = self.output_layer.nodes

        res = np.array([node.value for node in output_nodes]).transpose(1, 0, 2)
        errors = dmse(res, y_test)
        for t in reversed(range(seq_len - 1)):
            # Градиент ошибки для узла на текущем шаге
            output_nodes[t].grad = errors[:, t]
        output_nodes[-1].grad, output_nodes[-1].ref_count = np.zeros_like(errors[:, -1]), 1
        output_nodes[-1].backward(errors[:, -1])
        for i, layer in enumerate(reversed(self.layers)):
            layer.update_params(learning_rate=self.learning_rate)

    def fit(self, x: np.ndarray, y: np.ndarray, epoch_count: int = 5, batch_size: int = 32):
        """

        :param x: датасет формата (n, seq_len, x_in)
        :param y: ожидаемый результат формата (n, seq_len, y_out)
        :param epoch_count:
        :param batch_size:
        :return:
        """
        losses = []
        dataset = list(zip(x, y))
        for epoch in tqdm(range(epoch_count)):
            rng.shuffle(dataset)
            loss_ep = 0
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                self.forward(X_batch)
                # print(i, self.output_layer.output_signal.max())
                self.backward(y_batch)
            loss_ep = mse(self.predict(x), y)
            losses.append(loss_ep)
        return losses

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]


def test_init():
    in_len = 5
    batch = 6
    seq_len = 3
    out_len = 3
    inp = rng.random((batch, seq_len, in_len))
    net_arch = [inp.shape[-1], 4, out_len]
    test = RecurentNeuralNetworkGraph(net_arch, seq_len, 0.01)


def test_fit():
    # Параметры
    seq_len = 20  # длина последовательности
    k = 31
    part = 32
    # Генерация данных
    X = np.linspace(0, k * math.pi, k * (part) + 1)
    y = np.sin(X)

    X_seq, y_seq = create_sequences(y, seq_len)
    arch = [1, 50, 1]
    rnn = RecurentNeuralNetworkGraph(arch, RecurrentLayerTypes.RNN, seq_len, 0.001)
    loss = rnn.fit(X_seq.reshape((-1, seq_len, 1)), y_seq.reshape((-1, seq_len, 1)), epoch_count=5, batch_size=1)
    # plt.show()
    print(f"{np.array(loss)}")


# Создание выборок
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+1:i+seq_len+1])
    return np.array(X), np.array(y)


from RNN import get_test_inp, get_test_out
def test_mini():
    inp_len = 5
    out_len1 = 3
    out_len2 = 2
    seq_len = 4
    inp = get_test_inp()
    outp = get_test_out()
    test = RecurentNeuralNetworkGraph([inp_len, out_len1, out_len2],
                                      RecurrentLayerTypes.GRUgr,
                                      seq_len, 1e-1)
    # print(f'{test.output_layer.weights_node.value=}')
    loss = test.fit(inp, outp, epoch_count=2)
    print(loss)
"""test.output_layer.weights_node.value=array([[-0.61120268, -0.02611936],
       [-0.2681479 ,  0.11507216],
       [-0.29015443,  0.103804  ],
       [ 0.2647263 ,  0.65609751],
       [-0.86375996,  0.46433365]])
[0.4937747707702555, 0.31139967734011625]"""

import cProfile

def test_profile():
    cProfile.run("RecurentNeuralNetworkGraph.test_mini()", sort="cumtime")
