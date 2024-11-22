import math

from RecurrentLayerTypes import RecurrentLayerTypes, create_layer
from Layer import Layer

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


class RecurentNeuralNetwork:
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

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Прямой проход
        :param input_data: входящие данные формата (l, seq_len, in)
        :return: result формата (l, seq_len, out)
        """
        signal = input_data
        for layer in self.layers:
            signal = layer.forward(signal)
        return signal.copy()

    def backward(self, y_test: np.array) -> None:
        # out_s (l ,out), y_test (batch, out) -> (l, out)
        error_signal = dmse(self.output_layer.output_signal, y_test)
        for i, layer in enumerate(reversed(self.layers)):
            error_signal = layer.backward(error_signal, is_need_out=i != (len(self.layers) - 1))
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
            loss_ep =  mse(self.forward(x), y)
            losses.append(loss_ep)
        return losses

    def fit_with_plt(self, x: np.ndarray, y: np.ndarray, epoch_count: int = 5, batch_size: int = 32):
        """

        :param x: датасет формата (n, seq_len, x_in)
        :param y: ожидаемый результат формата (n, seq_len, y_out)
        :param epoch_count:
        :param batch_size:
        :return:
        """
        losses = []
        dataset = list(zip(x, y))
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'r-', lw=2)
        ax.set_xlim(0, epoch_count)
        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss in Real-Time')
        for epoch in tqdm(range(epoch_count)):
            rng.shuffle(dataset)
            loss_ep = 0
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                self.forward(X_batch)
                self.backward(y_batch)
            loss_ep =  mse(self.forward(x), y)
            losses.append(loss_ep)
            line.set_data(range(len(losses)), losses)
            # Обновляем оси графика
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)
        plt.ioff()
        plt.show()
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
    test = RecurentNeuralNetwork(net_arch, seq_len, 0.01)


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
    rnn = RecurentNeuralNetwork(arch, seq_len, 0.001)
    loss = rnn.fit(X_seq.reshape((-1, seq_len, 1)), y_seq.reshape((-1, seq_len, 1)), epoch_count=50, batch_size=1)
    plt.show()
    print(f"{np.array(loss)}")


# Создание выборок
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+1:i+seq_len+1])
    return np.array(X), np.array(y)
