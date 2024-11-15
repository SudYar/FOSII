import numpy as np

from Layer import Layer
from Functions import Functions, rng, dmse, mse


class Dense(Layer):
    # weights.shape = (input_size; output_size)
    weights: np.ndarray
    # bias.shape = (output_size)
    bias: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    output_signal: np.ndarray
    summator: np.ndarray
    input_signal: np.ndarray
    input_size: int
    output_size: int

    def __init__(self, input_size: int, output_size: int, activation_function: Functions):
        self.activation_function = activation_function
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self._init_weights(input_size, output_size)
        self.bias = self._init_biases(output_size)

    def _init_weights(self, input_size, output_size):
        limit = np.sqrt(6. / (input_size + output_size))
        return np.array(rng.uniform(-limit, limit + 1e-5, size=(input_size, output_size)))

    def _init_biases(self, output_size):
        return np.array(rng.random(output_size) * 2 - 1)

    def forward(self, signal: np.ndarray):
        if signal.shape[1] != self.input_size:
            raise Exception(f"Размерность сигнала{signal.shape[1]} отличается от " +
                            f"размерности входящего слоя: {self.input_size}")
        self.output_signal = np.array([[]])
        self.input_signal = signal
        #         in_sig(batch, in), w (in, out), + b (out ,)
        self.summator = np.matmul(self.input_signal, self.weights) + self.bias
        self.output_signal = self.activation_function.calc(self.summator)
        return self.output_signal

    def backward(self, error_signal: np.ndarray, is_need_out=True) -> np.ndarray | None:
        """

        :param error_signal: переданный дельта с предыдущих слоев (размерности (batch, out))
        :param is_need_out: Необходимость проталкивать delta дальше. Если это первый слой, то нет необходимости
        :return:
        """
        # f (batch, out) * (batch, out)
        delta = self.activation_function.derivative(self.summator) * error_signal
        # (batch, in).T x (batch, out) -> (in, out)
        self.dW = np.matmul(self.input_signal.T, delta)
        self.db = np.sum(delta, axis=0)
        if is_need_out:
            # [error(batch, out)x w(in, out).T  ] -> (batch, in)
            delta = np.matmul(delta, self.weights.T)
        return delta

    def update_params(self, learning_rate: float | None):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db


def test_forward():
    test = Dense(5, 3, Functions.none)
    print(test.weights)
    print(test.bias)
    inp = np.array([[1, 0, 1, 0, 1]]).reshape(1, 5)
    result = test.forward(inp)
    print(result)

    """
    [[ 0.76634006 -0.33404309 -0.42451161 -0.48043658  0.49572811]
 [-0.49067389  0.5295419   0.40298306 -0.13548592 -0.01413979]
 [ 0.70883497  0.50398643 -0.59962183  0.43127379 -0.16798996]]
[[ 0.98281432]
 [-0.57678455]
 [-0.17904883]]
[[ 1.82037088]
 [-0.67861517]
 [-0.23782566]]"""


def test_backprop():
    test = Dense(5, 3, Functions.none)
    print(f"init_weights: {test.weights}")
    print(f"init_bias: {test.bias}")
    inp = np.array([[1, 0, 1, 0, 1]]).reshape(1, 5)
    result = test.forward(inp)
    print(f"result {result}")
    out_s = np.array([0, 0, 1]).reshape(1, 3)
    dmse_s = dmse(result, out_s)
    loss = test.backward(dmse_s)
    test.update_params(0.01)
    result2 = test.forward(inp)
    print(f"result2: {result2}")
    print(f"weights_after_back:{test.weights}")
    print(f"dW: {test.dW}")
    """init_weights: [[ 0.76634006 -0.33404309 -0.42451161], 
    [-0.48043658  0.49572811 -0.49067389], 
    [ 0.5295419   0.40298306 -0.13548592], 
    [-0.01413979  0.70883497  0.50398643], 
    [-0.59962183  0.43127379 -0.16798996]]
init_bias: [ 0.98281432 -0.57678455 -0.17904883]
result [[ 1.67907444 -0.07657079 -0.90703633]]
result2: [[ 1.54474849 -0.07044513 -0.75447342]]
weights_after_back:[[ 0.73275857 -0.33251167 -0.38637089],
    [-0.48043658  0.49572811 -0.49067389],
    [ 0.49596041  0.40451447 -0.09734519],
    [-0.01413979  0.70883497  0.50398643],
    [-0.63320332  0.4328052  -0.12984924]]
dW: [[ 3.35814889 -0.15314158 -3.81407266], 
    [ 0.          0.          0.        ], 
    [ 3.35814889 -0.15314158 -3.81407266], 
    [ 0.          0.          0.        ], 
    [ 3.35814889 -0.15314158 -3.81407266]]"""

def test_bacprop_2layers():
    lay1 = Dense(5, 4, Functions.sigmoid)
    lay2 = Dense(4, 3, Functions.none)
    print(f"init_weights: {lay1.weights}")
    print(f"init_weights: {lay2.weights}")
    print(f"bias1: {lay1.bias};\n bias2: {lay2.bias}")

    inp = np.array([[1, 0, 1, 0, 1]]).reshape(1, 5)
    result1 = lay1.forward(inp)
    result = lay2.forward(result1)
    print(f"result {result}")
    out_s = np.array([0, 0, 1]).reshape(1, 3)
    dmse_s = dmse(result, out_s)
    delta2 = lay2.backward(dmse_s)
    lay1.backward(delta2, is_need_out=False)

    lay1.update_params(0.01)
    lay2.update_params(0.01)

    res1 = lay1.forward(inp)
    res = lay2.forward(res1)
    print(f"result2: {res}")
    print(f"weights_after_back:{lay1.weights}, \n lay2_weigth: {lay2.weights}")
    print(f"dW1: {lay1.dW},\n dW2: {lay2.dW}")

    """
    init_weights: [[ 0.72251287 -0.31493867 -0.40023324 -0.45295983  0.4673774 ]
 [-0.46261166  0.49925735  0.37993649 -0.12773711 -0.01333084]
 [ 0.66829654  0.47516341 -0.56532879  0.40660925 -0.15838223]
 [ 0.80247445 -0.47094049 -0.14618866  0.72389501  0.13097715]]
init_weights: [[-0.74982772 -0.25898291 -0.75532695 -0.61120268]
 [-0.02611936 -0.2681479   0.11507216 -0.29015443]
 [ 0.103804    0.2647263   0.65609751 -0.86375996]]
bias1: [[ 0.47150135]
 [ 0.20950201]
 [-0.53731956]
 [-0.89091086]];
 bias2: [[ 0.50152955]
 [-0.25262151]
 [-0.50155969]]
result [[-0.77826755 -0.51124748 -0.45674897]]
result2: [[-0.73841389 -0.48552645 -0.38247211]]
weights_after_back:[[ 0.72097936 -0.31493867 -0.40176676 -0.45295983  0.46584388]
 [-0.46237745  0.49925735  0.38017069 -0.12773711 -0.01309664]
 [ 0.67025334  0.47516341 -0.56337199  0.40660925 -0.15642542]
 [ 0.79308817 -0.47094049 -0.15557493  0.72389501  0.12159088]], 
 lay2_weigth: [[-0.7376988  -0.25075905 -0.74978556 -0.60382298]
 [-0.01815181 -0.26274562  0.11871232 -0.28530667]
 [ 0.12650672  0.28011958  0.66646979 -0.84994674]]
dW1: [[ 0.15335161  0.          0.15335161  0.          0.15335161]
 [-0.02342042  0.         -0.02342042  0.         -0.02342042]
 [-0.19568023  0.         -0.19568023  0.         -0.19568023]
 [ 0.93862754  0.          0.93862754  0.          0.93862754]],
 dW2: [[-1.21289177 -0.82238533 -0.55413861 -0.73797072]
 [-0.7967541  -0.54022865 -0.36401616 -0.48477631]
 [-2.27027175 -1.53932793 -1.03722795 -1.38132199]]"""

    """dmse=[[-1.55653512], [-1.02249496], [-2.91349795]]"""

    """delta from lay2:[[-1.55653512], [-1.02249496], [-2.91349795]]"""
    """delta from lay1:[[ 0.15335161], [-0.02342041], [-0.19568023], [ 0.93862754]]"""
