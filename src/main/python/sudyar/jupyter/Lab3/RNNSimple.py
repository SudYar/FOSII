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
    inp = get_test_inp()
    out = get_test_out()
    test1 = RNNSimple((seq_len, inp_len), out_len, Functions.tanh)

    res1 = test1.forward(inp)
    assert np.allclose(res1, get_expected_result())
    errors = dmse(out, res1)
    grad = test1.backward(errors, is_need_out=True)
    assert np.allclose(grad, get_expected_grad())
    print(test1.db)
    print(test1.dW)

def test_2layers():
    inp_len = 5
    out_len1 = 3
    out_len2 = 2
    seq_len = 4
    inp = get_test_inp()
    out = get_test_out()

    test1 = RNNSimple((seq_len, inp_len), out_len1, Functions.tanh)
    test2 = RNNSimple((seq_len, out_len1), out_len2, Functions.tanh)

    res1 = test1.forward(inp)
    res2 = test2.forward(res1)
    errors = dmse(res2, out)
    grad2 = test2.backward(errors, is_need_out=True)
    grad1 = test1.backward(grad2, is_need_out=True)
    test2.update_params(learning_rate=1e-1)
    test1.update_params(learning_rate=1e-1)
    print(grad1)
    print(test1.db)
    print(test2.db)
    print(test1.bias)
    print(test2.bias)
    print(f"{test1.weights=}")

    res1 = test1.forward(inp)
    res2 = test2.forward(res1)
    errors = dmse(res2, out)
    grad2 = test2.backward(errors, is_need_out=True)
    grad1 = test1.backward(grad2, is_need_out=True)
    test2.update_params(learning_rate=1e-1)
    test1.update_params(learning_rate=1e-1)
    print(grad1)
    print(test1.db)
    print(test2.db)
    print(test1.bias)
    print(test2.bias)
"""[[[-0.01004544  0.02561033  0.05805823  0.07614478  0.01870129]
  [-0.01563835 -0.04695479  0.02481172  0.06802775 -0.02593562]
  [ 0.00259628  0.00979623  0.04341116  0.04895149  0.00429322]
  [-0.02244236 -0.01236237  0.02124064  0.05784361 -0.00077753]]

 [[-0.05452442  0.02984299 -0.06148843 -0.02788184  0.04048273]
  [-0.01822631 -0.00216116 -0.02101594 -0.0052062   0.00559492]
  [-0.0022471   0.03069597 -0.00481534 -0.01550506  0.02103383]
  [-0.01219615 -0.01131923  0.01832772  0.04192941 -0.00362533]]

 [[-0.10491752  0.10735884 -0.01978165  0.05348863  0.10786445]
  [ 0.0316017  -0.05443376  0.07487013  0.08092993 -0.04888242]
  [-0.00937485 -0.0210219  -0.00047827  0.01830748 -0.01045219]
  [-0.01997634  0.01754269 -0.02270279 -0.0130029   0.01916596]]

 [[-0.07122885  0.10367448  0.04214874  0.09572756  0.09185023]
  [-0.04679422 -0.0222998   0.05548344  0.13363296  0.0003446 ]
  [ 0.04232418 -0.02802967  0.07891717  0.06356888 -0.03547904]
  [ 0.00575631 -0.03055162  0.02188846  0.03333715 -0.02265173]]

 [[-0.04231592  0.02618314 -0.04638005 -0.02108982  0.03336072]
  [-0.0092798   0.01788539 -0.0053726  -0.00317389  0.01513701]
  [ 0.01670405  0.04070254  0.08080653  0.06876959  0.01852394]
  [ 0.05000223 -0.06870454  0.08786259  0.08200762 -0.06507922]]]
  
  [0.28391713 1.0341398  0.50045172]
[-1.33595736  6.25944188]
[-0.83829924 -0.38315138 -0.86589251]
[-0.11902578 -1.12750388]
test1.weights=array([[ 0.64115891, -0.3305161 , -0.38091501],
       [-0.4088126 ,  0.38736197, -0.43366494],
       [ 0.45828246,  0.31639156, -0.12674838],
       [-0.03323854,  0.54504438,  0.40159632],
       [-0.52105913,  0.33895693, -0.16440975],
       [ 0.76430008, -0.37694334, -0.10083733],
       [ 0.62122465,  0.07497924,  0.32311115],
       [ 0.18086381, -0.36970132, -0.62486822]])
    
[[[ 2.69935921e-02 -2.76326525e-02  5.50704545e-02  4.01163173e-02
   -3.22691864e-02]
  [-7.64915835e-03 -1.96855682e-02  1.09944562e-02  2.61384475e-02
   -1.20022927e-02]
  [ 1.28493423e-02 -2.11029863e-02  2.45557715e-02  2.00034333e-02
   -2.06253739e-02]
  [ 3.78032747e-03 -2.78002763e-02  1.62855934e-03  7.19066498e-03
   -2.02692471e-02]]

 [[ 5.14460488e-02 -3.76739663e-02  4.61042875e-02  8.88524657e-03
   -4.59708044e-02]
  [-6.50126702e-04 -5.35575364e-03  1.29565165e-03  3.84222791e-03
   -3.56113723e-03]
  [-6.31552438e-05 -4.87049001e-03  2.68466766e-03  4.55327293e-03
   -3.54032860e-03]
  [ 6.70467949e-03 -2.00042886e-02  4.20752446e-03  4.31380156e-03
   -1.61097709e-02]]

 [[ 6.14729634e-04 -2.02125178e-04  3.58904032e-04 -1.81151796e-04
   -3.63000760e-04]
  [ 1.18560937e-02 -1.71811128e-02  2.51013312e-02  2.02845375e-02
   -1.76916859e-02]
  [ 9.38799576e-04 -1.62820771e-02  7.10867730e-03  1.20593619e-02
   -1.20274767e-02]
  [ 9.61025524e-03 -1.42872254e-02  6.21067339e-03  1.54255241e-03
   -1.33083961e-02]]

 [[ 7.49337400e-02 -7.10975869e-02  8.25261174e-02  3.47258256e-02
   -7.93988093e-02]
  [-9.27396461e-03 -1.09984765e-02  8.60341438e-03  2.23463611e-02
   -5.35922796e-03]
  [ 8.76391910e-03 -8.55133167e-03  2.04464789e-02  1.56109710e-02
   -1.04226200e-02]
  [ 2.18028748e-03 -1.77236553e-02  4.47439595e-03  8.47119348e-03
   -1.31608336e-02]]

 [[ 1.67763437e-02 -1.68202916e-02  1.60218095e-02  5.46974043e-03
   -1.81690847e-02]
  [ 4.88585811e-05 -2.68672962e-03  2.03686629e-03  3.01662017e-03
   -2.02947576e-03]
  [ 4.46358377e-03 -2.87677151e-06  2.36630099e-02  2.05678053e-02
   -3.54133038e-03]
  [ 1.80132402e-02 -2.92083178e-02  1.76954125e-02  1.01402633e-02
   -2.71464425e-02]]]
[0.72357195 0.24919234 0.39175428]
[-1.62321697  0.8725348 ]
[-0.91065644 -0.40807061 -0.90506794]
[ 0.04329592 -1.21475736]"""

def get_test_inp():
    batch_size = 5
    seq_len = 4
    inp_len = 5
    # out_len = 2
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