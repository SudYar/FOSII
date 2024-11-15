from LayerTypes import LayerTypes, create_layer
from Layer import Layer

import numpy as np
from tqdm import tqdm
from Functions import mse, dmse, Functions, rng


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])


class NeuralNetwork:
    layers: list[Layer]
    learning_rate: float

    def __init__(self, layers: list[Layer], learning_rate: float):
        if len(layers) == 0:
            raise Exception("Model is empty")

        self.layers = list(layers)
        self.learning_rate = learning_rate

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Прямой проход
        :param input_data: входящие данные формата (l, in)
        :return: result формата (l, out)
        """
        signal = input_data
        for layer in self.layers:
            signal = layer.forward(signal)
        return signal.copy()

    def backward(self, y_test: np.array) -> None:
        # if np.any(y_test.shape != self.output_layer.output_signal.shape):
        #     raise Exception(f"Invalid shape of y_test. Use {self.output_layer.output_signal.shape = }", )
        # out_s (l ,out), y_test (batch, out) -> (l, out)
        error_signal = dmse(self.output_layer.output_signal, y_test)
        for i, layer in enumerate(reversed(self.layers)):
            error_signal = layer.backward(error_signal, is_need_out=i != (len(self.layers) - 1))
            layer.update_params(learning_rate=self.learning_rate)

    def fit(self, x: np.ndarray, y: np.ndarray, epoch_count: int = 5, batch_size: int = 32):
        losses = np.zeros(shape=epoch_count)
        dataset = list(zip(x, y))
        for epoch in tqdm(range(epoch_count)):
            rng.shuffle(dataset)
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                y_predicted = self.forward(X_batch)
                losses[epoch] += np.sum(mse(y_predicted, y_batch))
                self.backward(y_batch)
        losses /= len(x)
        return losses

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]


def test_init():
    inp = np.array([[1, 0, 1, 0, 1]]).reshape(1, 5)
    net_arch = (inp.shape[1], 4, 3)
    test = NeuralNetwork([
        create_layer(LayerTypes.DENSE, input_size=net_arch[0],
                     output_size=net_arch[1], activation_function=Functions.sigmoid),
        create_layer(LayerTypes.DENSE, input_size=net_arch[1],
                     output_size=net_arch[2], activation_function=Functions.none)
    ], 0.01)
    """
    """

def test_init2():
    net_arch_mnist = ((20, 5, 5), (2, 12, 12), (10, 3, 3), (2, 5, 5), 20, 10)
    NN = NeuralNetwork([
        create_layer(LayerTypes.CONVOLUTIONAL, kernel_count=net_arch_mnist[0][0],
                     kernel_size=(5, 5), input_size=(1, 28, 28),
                     activation_function=Functions.relu),
        create_layer(LayerTypes.POOLING, pool_size=2),
        create_layer(LayerTypes.CONVOLUTIONAL, kernel_count=net_arch_mnist[2][0],
                     kernel_size=(3, 3), input_size=(net_arch_mnist[0][0], 12, 12),
                     activation_function=Functions.relu),
        create_layer(LayerTypes.POOLING, pool_size=2),
        create_layer(LayerTypes.FLATTEN),
        create_layer(LayerTypes.DENSE, input_size=250, output_size=20, activation_function=Functions.tanh),
        create_layer(LayerTypes.DENSE, input_size=20, output_size=10, activation_function=Functions.none)
    ], 0.07)

    inp = rng.random((2,1,28,28))
    outp = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print(outp.shape)
    print(NN.fit(inp, outp, 2))

def test_fit():
    inp = np.array([[1, 0, 1, 0, 1]]).reshape(1, 5)
    outp = np.array([0, 0, 1]).reshape(1, 3)
    net_arch = (inp.shape[1], 4, outp.shape[1])
    test = NeuralNetwork([
        create_layer(LayerTypes.DENSE, input_size=net_arch[0],
                     output_size=net_arch[1], activation_function=Functions.sigmoid),
        create_layer(LayerTypes.DENSE, input_size=net_arch[1],
                     output_size=net_arch[2], activation_function=Functions.none)
    ], 0.01)
    for lay in test.layers:
        print(f"({lay.input_size}, {lay.output_size}): {lay.weights}")
    test.fit(inp, outp, epoch_count=1)
    print("after fit")
    for lay in test.layers:
        print(f"({lay.input_size}, {lay.output_size}): {lay.weights}")

#     todo проверить размерности входящих данных для forward
