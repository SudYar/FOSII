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
    inp_len = 5
    out_len = 2
    seq_len = 4
    # inp = rng.random((5, seq_len, inp_len))
    # out = rng.random((5, seq_len, out_len))
    inp = get_test_inp()
    out = get_test_out()
    test1 = GRULayer((seq_len, inp_len), out_len, Functions.tanh)

    res1 = test1.forward(inp)
    errors = dmse(res1, out)
    print(f"{errors=}")
    test1.backward(errors, is_need_out=True)
    test1.update_params(learning_rate=1e-1)

    print(f"{test1.db_z=}")
    res1 = test1.forward(inp)
    errors = dmse(res1, out)
    print(f"{errors=}")
    """errors=array([[[ 0.54084139, -0.13489263],
        [ 0.74931525, -0.25787829],
        [ 0.5442109 , -0.23801909],
        [ 0.20530697, -0.66595023]],

       [[ 0.26433387, -0.0743128 ],
        [ 0.04531307, -0.42001366],
        [ 0.1065362 , -0.2522244 ],
        [ 0.18290737, -0.6106005 ]],

       [[ 0.0624503 , -0.1714808 ],
        [ 0.71600176, -0.17950038],
        [ 0.26926   , -0.34905088],
        [ 0.03712435, -0.22164249]],

       [[ 0.43987145, -0.369261  ],
        [ 0.63143393, -0.55200086],
        [ 0.54253603,  0.0310024 ],
        [ 0.31954289, -0.40195301]],

       [[ 0.21336478, -0.26106756],
        [ 0.03909029, -0.44480206],
        [ 0.18684594,  0.12340596],
        [ 0.55588862, -0.39424471]]])
test1.db_z=array([0.22656952, 1.26172144])
errors=array([[[ 0.50763453, -0.10902406],
        [ 0.70210674, -0.22139348],
        [ 0.47026643, -0.20496348],
        [ 0.13181692, -0.63743809]],

       [[ 0.20213507, -0.05802577],
        [-0.02468922, -0.38587631],
        [ 0.04974981, -0.20812963],
        [ 0.08969718, -0.56759001]],

       [[ 0.00373912, -0.15187044],
        [ 0.64181658, -0.14180297],
        [ 0.21456203, -0.3040416 ],
        [-0.06146463, -0.17845826]],

       [[ 0.39421178, -0.34909072],
        [ 0.56910159, -0.51889369],
        [ 0.45783338,  0.0674172 ],
        [ 0.24668856, -0.36541491]],

       [[ 0.17354453, -0.23873402],
        [-0.00842625, -0.40274182],
        [ 0.10793415,  0.16353494],
        [ 0.47916429, -0.35469156]]])"""


from src.main.python.sudyar.jupyter.Lab3.RNN import get_test_inp, get_test_out
def test_2layers():
    inp_len = 5
    out_len1 = 3
    out_len2 = 2
    seq_len = 4
    inp = get_test_inp()
    out = get_test_out()

    test1 = GRULayer((seq_len, inp_len), out_len1, Functions.tanh)
    test2 = GRULayer((seq_len, out_len1), out_len2, Functions.tanh)
    print(f"{test2.candidate_weights=}")

    res1 = test1.forward(inp)
    res2 = test2.forward(res1)
    print(f"{res1=}")
    errors = dmse(res2, out)
    loss = mse(res2, out)
    print(f"{errors=},\n{loss=}")
    grad2 = test2.backward(errors, is_need_out=True)
    grad1 = test1.backward(grad2, is_need_out=True)
    test2.update_params(learning_rate=1e-1)
    test1.update_params(learning_rate=1e-1)
    print(f"{test1.db_z=}")
    print(f"{test1.db_h=}")
    print(f"{grad1=}")
    print(f"{test1.bias_z=}")
    print(f"{test1.bias_h=}")
    print(f"{test1.candidate_weights=}")

    res1 = test1.forward(inp)
    res2 = test2.forward(res1)
    errors2 = dmse(res2, out)
    loss2 = mse(res2, out)
    print(f"{loss2=}")
    grad2 = test2.backward(errors2, is_need_out=True)
    grad1 = test1.backward(grad2, is_need_out=True)
    test2.update_params(learning_rate=1e-1)
    test1.update_params(learning_rate=1e-1)
    print(test1.db_z)
    print(test1.db_h)
    print(grad1)
    print(test1.bias_z)
    print(test1.bias_r)
    print(test1.bias_h)
    """test2.candidate_weights=array([[-0.82131469,  0.81332427],
       [-0.31867834, -0.29537062],
       [-0.24537188, -0.3998399 ],
       [-0.76294857,  0.12309546],
       [-0.76978703, -0.19052315]])
res1=array([[[-0.22285897, -0.76055051,  0.10222804],
        [-0.16038198, -0.68021862,  0.11131515],
        [-0.21958132, -0.44273948,  0.03882356],
        [-0.24566651, -0.44237537, -0.12921149]],

       [[-0.49084567, -0.51691977,  0.0503996 ],
        [-0.41671733, -0.65885601, -0.12069496],
        [-0.46291922, -0.79650413,  0.42963388],
        [-0.49404356, -0.6627225 ,  0.19660981]],

       [[-0.55946671, -0.49520396, -0.06759247],
        [-0.56087681, -0.68012684,  0.03025894],
        [-0.47180679, -0.72023337, -0.07111286],
        [-0.5632581 , -0.51705465, -0.20855514]],

       [[-0.31473499, -0.43213105,  0.01878118],
        [-0.27784487, -0.66352383,  0.07485267],
        [-0.42067857, -0.62258374, -0.28319026],
        [-0.49955199, -0.66089921,  0.14413122]],

       [[-0.33112478, -0.75108639,  0.25891677],
        [-0.37320109, -0.85557568,  0.47929289],
        [-0.30463637, -0.49914788,  0.01921213],
        [-0.48891266, -0.55899054, -0.24660276]]])
res2=array([[[-0.23989162,  0.29459866],
        [-0.39483912,  0.42436621],
        [-0.46839939,  0.45251419],
        [-0.49102766,  0.48185183]],

       [[-0.14326838,  0.18608   ],
        [-0.2268531 ,  0.33583772],
        [-0.31171563,  0.30946536],
        [-0.32066514,  0.30617372]],

       [[-0.1059932 ,  0.18126165],
        [-0.1776312 ,  0.25941315],
        [-0.23379473,  0.34752387],
        [-0.25351193,  0.35276   ]],

       [[-0.19181459,  0.24759938],
        [-0.32575789,  0.37356738],
        [-0.33083706,  0.4360804 ],
        [-0.33725592,  0.3759189 ]],

       [[-0.23113793,  0.22996193],
        [-0.33783772,  0.28478288],
        [-0.39552788,  0.36779629],
        [-0.35965743,  0.39656291]]])
errors=array([[[ 0.14404335,  0.11783947],
        [ 0.24206435,  0.16974648],
        [ 0.05264024,  0.26100568],
        [-0.27641106, -0.12725927]],

       [[-0.05730735,  0.074432  ],
        [-0.33074124, -0.10566491],
        [-0.36468625,  0.12378614],
        [-0.20826605, -0.19753051]],

       [[-0.28239728, -0.00749534],
        [ 0.32894752,  0.10376526],
        [-0.17351789,  0.05900955],
        [-0.34140477,  0.221104  ]],

       [[ 0.08327416, -0.14096025],
        [ 0.18969685, -0.17057305],
        [ 0.10766518,  0.49443216],
        [-0.13490237,  0.07036756]],

       [[-0.17245517, -0.06801523],
        [-0.45513509, -0.12608685],
        [-0.23821115,  0.54711852],
        [ 0.09613703,  0.07862516]]])
loss=0.31928914109065687
test1.db_z=array([-0.17675427, -0.00603016,  0.03978545])
test1.db_h=array([ 0.98737905,  0.03710405, -0.05763355])
grad1=array([[[ 4.51797596e-02, -3.05220479e-02, -6.07538821e-02,
         -3.92334635e-02,  1.86636794e-02],
        [ 1.29188869e-02, -1.75494249e-02, -4.71745742e-02,
         -1.81341459e-02,  2.17518946e-02],
        [-3.34363207e-02, -1.30895558e-02, -3.20476801e-02,
         -3.14820395e-03,  1.83245194e-02],
        [-2.52278369e-02, -8.73836453e-03,  1.11288747e-02,
          3.24234579e-03, -1.12930261e-02]],

       [[-1.74439097e-01,  2.33742703e-02, -4.05505634e-02,
          3.47365121e-02,  5.49473712e-02],
        [-9.48919717e-02, -1.69812163e-02, -4.83083103e-03,
          1.39301032e-02,  6.10606227e-03],
        [-6.46335936e-02, -2.06674488e-02, -1.40222299e-02,
          1.27741161e-02,  2.97442247e-03],
        [-1.67225125e-02, -2.38201873e-03,  2.13248610e-02,
          3.53829459e-03, -1.47438671e-02]],

       [[-6.98438215e-02,  3.24704529e-03, -4.93486324e-02,
          1.98771533e-03,  4.08135711e-02],
        [-3.11636689e-02, -2.15750845e-02, -5.06953475e-02,
         -9.35818174e-03,  2.70148253e-02],
        [-5.39819050e-02, -3.28700486e-02, -4.35823609e-02,
         -3.66127754e-03,  1.93510698e-02],
        [-3.07581687e-02, -1.55943239e-02, -3.79701649e-02,
         -6.27266557e-03,  1.99611825e-02]],

       [[ 6.12970522e-02, -1.56307339e-02, -3.91007314e-04,
         -2.19550008e-02, -1.66815165e-02],
        [ 3.51098340e-02, -2.03820319e-02, -5.12798919e-02,
         -2.60916065e-02,  2.03920495e-02],
        [-3.19091976e-02, -1.52318664e-02, -7.58563273e-02,
         -6.99907905e-03,  4.73588468e-02],
        [-1.31556696e-02, -7.54248065e-03, -1.25186925e-02,
         -2.88880775e-03,  5.13751385e-03]],

       [[-1.73742304e-01,  5.25959700e-03, -8.89358849e-02,
          2.21040026e-02,  7.91608088e-02],
        [-7.49888305e-02, -3.09405310e-02, -7.19913681e-02,
         -9.84057032e-03,  3.90913473e-02],
        [-2.74583599e-02, -3.64561290e-02, -8.99575787e-02,
         -1.13182569e-02,  4.53388419e-02],
        [ 5.27614772e-03,  3.64096698e-03, -7.89101802e-03,
         -5.15110573e-05,  7.79771088e-03]]])
test1.bias_z=array([0.25557668, 0.99779375, 0.79667574])
test1.bias_h=array([-0.14391222, -0.14781561, -0.33885566])
test1.candidate_weights=array([[-0.69610192, -0.08934273, -0.70365655],
       [-0.33315227, -0.69943226,  0.46738459],
       [-0.66875442,  0.07473984,  0.685581  ],
       [-0.0923386 , -0.281341  ,  0.53785779],
       [ 0.30687668, -0.53592769, -0.09805431],
       [-0.20473742,  0.44909846, -0.01405349],
       [-0.09768706, -0.11768541,  0.32622747],
       [ 0.69985785,  0.20167318, -0.10940494]])
errors-errors2=array([[[-0.05082366,  0.03649228],
        [-0.07844026,  0.05396065],
        [-0.08841332,  0.06434004],
        [-0.09533534,  0.06752704]],

       [[-0.03857651,  0.03699286],
        [-0.06791158,  0.05370321],
        [-0.09537181,  0.07036049],
        [-0.10312351,  0.07663684]],

       [[-0.03536584,  0.03839675],
        [-0.06942088,  0.05945211],
        [-0.09197013,  0.07102167],
        [-0.09965592,  0.07442219]],

       [[-0.03527495,  0.03662035],
        [-0.07063467,  0.05487474],
        [-0.08719047,  0.06359589],
        [-0.10006214,  0.07502407]],

       [[-0.05386943,  0.03797444],
        [-0.09092538,  0.06088814],
        [-0.0970821 ,  0.06631683],
        [-0.10231216,  0.07078504]]])
loss2=0.27044581497011144
[-0.05447655 -0.00898051  0.03382799]
[ 0.17496402  0.02814249 -0.09478068]
[[[ 1.04635254e-01 -1.25984495e-02 -9.20233915e-03 -3.05256304e-02
   -5.81854858e-03]
  [ 4.51163554e-02  2.93339138e-03 -2.32899463e-02 -1.46802155e-02
    1.42030321e-02]
  [-1.02152708e-02 -4.31055852e-03 -7.78463198e-03 -1.13121128e-03
    3.24384477e-03]
  [-1.59871229e-02 -2.05410064e-03  3.00364824e-02  7.03510766e-03
   -2.12840891e-02]]

 [[-1.03650818e-01 -2.38559140e-03 -8.37115349e-03  1.32788332e-02
    4.31941567e-03]
  [-6.30584734e-02 -1.66635298e-02  1.66371428e-02  6.31131694e-03
   -1.81390073e-02]
  [-4.02263464e-02 -1.63899251e-02  6.75289299e-03  1.19349360e-02
   -1.29905841e-02]
  [-6.98423174e-03  6.48674477e-03  3.61749505e-02  7.62084537e-03
   -1.97008389e-02]]

 [[-1.72442436e-02 -1.68229476e-03 -1.10521863e-02 -1.51857576e-03
    6.13610898e-03]
  [-2.88375137e-03 -3.00678242e-03 -2.07526055e-02 -4.77836135e-03
    1.28014471e-02]
  [-3.32398320e-02 -2.62682424e-02 -2.52660729e-02 -6.88480242e-03
    3.42374272e-03]
  [-1.82355106e-02 -1.54419632e-02 -2.52103447e-02 -7.56664069e-03
    8.02024132e-03]]

 [[ 1.30171819e-01  1.59621895e-02  6.71319672e-02 -3.10936830e-03
   -4.05417592e-02]
  [ 6.43063751e-02  8.69544650e-03 -1.71631331e-02 -1.51198117e-02
    1.32000464e-02]
  [-8.13838588e-03 -5.56671792e-03 -5.70647712e-02 -6.20006113e-03
    3.57043108e-02]
  [-3.44505579e-03 -2.15546075e-03 -3.07994155e-05 -4.13964216e-05
   -1.26006805e-03]]

 [[-9.78745365e-02 -2.13757332e-03 -4.85449829e-02  4.37436810e-03
    3.21368780e-02]
  [-3.82525345e-02 -2.32279114e-02 -4.53983574e-02 -1.46996352e-02
    1.68265800e-02]
  [-9.50985108e-03 -2.96589956e-02 -7.97781715e-02 -1.37635119e-02
    3.59684604e-02]
  [ 1.44524676e-02  1.28857398e-02  5.18634499e-03  4.54843450e-03
    4.69671879e-03]]]
[0.26102434 0.9986918  0.79329294]
[-0.16140862 -0.15062985 -0.32937759]"""