import numpy as np
import pandas

from Layer import Layer
from Functions import Functions, rng, dmse


def initialize_conv_weights(input_channels: int, kernel_size:tuple[int, int],
                            kernel_count: int, method='he') -> np.ndarray:
    kernel_height, kernel_width = kernel_size
    fan_in = input_channels * kernel_height * kernel_width
    fan_out = kernel_count * kernel_height * kernel_width

    if method == 'he':
        limit = np.sqrt(6 / fan_in)
    else:
        limit = np.sqrt(6 / (fan_in + fan_out))

    return np.array(rng.uniform(-limit, limit + 1e-5,
                                size=(kernel_count, input_channels, kernel_height, kernel_width)))

def init_bias(kernel_count:int) -> np.ndarray:
    return np.array(rng.random(kernel_count)*2-1)


def col_rot180(im2col: np.ndarray, channels: int, filer_length:int):
    """

    :param im2col: 2D матрица (channels_out, channels_it*kernel_size*kernel_size)
    :param channels:
    :param filer_length: для квадратной матрицы это kernel_size**2
    :return:
    """
    res = np.zeros_like(im2col)
    for i in range(channels):
        res[:, i*filer_length:(i+1)*filer_length] = np.fliplr(im2col[:, i*filer_length:(i+1)*filer_length])
    return res


class Convolution2D(Layer):

    summator: np.ndarray

    bias: np.ndarray
    dKernels: np.ndarray
    db: np.ndarray


    def __init__(self, kernel_count:int, kernel_size: tuple[int, int], input_size: tuple[int, int, int],
                 activation_function: Functions):
        """

        :param kernel_count:
        :param kernel_size:
        :param input_size: (channels, height, width)
        :param activation_function:
        """
        self.in_channel_count = input_size[0]
        self.out_channel_count = kernel_count
        self.kernel_count = kernel_count
        self.kernel_size = np.array(kernel_size, dtype=np.uint16)
        self.input_size = input_size
        self.output_size = np.array([input_size[0],self.in_channel_count, *(self.input_size[1:] - self.kernel_size + 1)]) # (batch_size,channels, height, width)
        self.kernels = initialize_conv_weights(self.in_channel_count, kernel_size, kernel_count)
        self.kernels_col = self.kernels.reshape(kernel_count, self.in_channel_count*kernel_size[0]*kernel_size[1]) # (k_count, channels*h*w)
        self.bias = init_bias(kernel_count)
        self.activation_function = activation_function


    def forward(self, signal: np.ndarray) -> np.ndarray:
        """

        :param signal: Входящий сигнал размерности (batch, channels, height, width)
        :return: output_signal выходящий сигнал размерности (batch, 'self.output_size')
        """
        self.input_signal = signal.copy()
        res = self.predict(signal)
        self.output_signal = res
        return res

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """

        :param signal: Входящий сигнал размерности (batch, channels, height, width)
        :return:
        """
        if signal[0].shape != self.input_size:
            raise Exception(f"Размерность сигнала {signal[0].shape} отличается от " +
                            f"размерности входящего слоя: {self.input_size}")

        batch_size, channel, input_height, input_width = signal.shape
        count_filters, _, filter_size, filter_size = self.kernels.shape

        #input_col (c*h*w ,batch_size * output_height * output_width)
        self.input_col = im2col(signal, filter_size)
        # summator (k, b*i*j) =        (k, c*h*w)   (c*h*w, b,i,j)                      (k, 1)
        self.summator = np.matmul(self.kernels_col, self.input_col) + self.bias.reshape(-1, 1)
        #   (b,k,i,j)
        self.summator = self.summator.reshape(-1, batch_size, self.output_size[2], self.output_size[3]).transpose(1, 0, 2, 3)
        self.output_signal = self.activation_function.calc(self.summator)

        return self.output_signal

    def backward(self, error_signal: np.ndarray, is_need_out=True) -> np.ndarray | None:
        """

                :param error_signal: переданный дельта с предыдущих слоев (размерности (batch, output_size)=bkij)
                :param is_need_out: Необходимость проталкивать delta дальше. Если это первый слой, то нет необходимости
                :return:
                """
        batch_size, delta_channel, delta_height, delta_width = error_signal.shape
        #                                       b,k,i,j * b,k,i,j
        delta = self.activation_function.derivative(self.summator) * error_signal
        #                       b  i  j  k
        delta = delta.transpose(0, 2, 3, 1).reshape(batch_size * delta_height * delta_width, delta_channel)
        #    (k, c*h*w) =       (c*h*w, b*i*j)X(b*i*j, k)
        self.dW_col = np.matmul(self.input_col, delta).transpose(1, 0)
        self.db = np.sum(delta, axis=0)

        if is_need_out:
            kernels_col90 = col_rot180(self.kernels_col, self.in_channel_count,  self.kernel_size.prod())
            #(b*i*j,c*h*w)
            new_delta = np.matmul(delta, kernels_col90)
            return col2im(new_delta, self.input_signal.shape, self.kernel_size[-1])

    def update_params(self, learning_rate: float):
        self.kernels_col -= learning_rate * self.dW_col
        self.bias -= learning_rate * self.db


def im2col(input_data, filter_size, stride=1, padding=0):
    batch_size, channel, input_height, input_width = input_data.shape
    output_height = (input_height + 2 * padding - filter_size) // stride + 1
    output_width = (input_width + 2 * padding - filter_size) // stride + 1
    # img = np.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    input_col = np.zeros((batch_size, channel, filter_size, filter_size, output_height, output_width))
    # for h in range(filter_size):
    #     height_max = h + stride * output_height
    #     for w in range(filter_size):
    #         width_max = w + stride * output_width
    #         input_col[:, :, h, w, :, :] = img[:, :, h:height_max:stride, w:width_max:stride]

    # input_col2 (batch, channels, i, j, h, w)
    input_col2 = np.lib.stride_tricks.sliding_window_view(input_data, (filter_size, filter_size), axis=(2, 3))
    # input_col = input_col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * output_height * output_width, -1)
    #                                 c  h  w  b  i  j
    input_col2 = input_col2.transpose(1, 4, 5, 0, 2, 3).reshape(-1, batch_size * output_height * output_width)

    return input_col2


def test_init():

    test = Convolution2D(3, (5, 5), (2, 7, 7), Functions.none)
    input_s = np.array([rng.random((2, 7, 7))*2])
    out_s = np.array([rng.random((3, 3, 3))])
    result = test.forward(input_s)
    print(result)

    result_loop = conv2d_with_loops(input_s, test.kernels, test.bias)
    # result_loop = Functions.relu.calc(result_loop)
    print(abs(result-result_loop).sum().sum())
    error = dmse(result, out_s)
    test_back = test.backward(error)


# skipy.ndimage




def col2im(input_col, input_shape, filter_size, stride=1, padding=0):
    batch_size, channel, input_height, input_width = input_shape
    output_height = (input_height + 2 * padding - filter_size) // stride + 1
    output_width = (input_width + 2 * padding - filter_size) // stride + 1
    #(b,c,h,w,i,j)
    input_col = input_col.reshape(batch_size, output_height, output_width, channel, filter_size, filter_size).transpose(
        0, 3, 4, 5, 1, 2)
    img = np.zeros(
        (batch_size, channel, input_height + 2 * padding + stride - 1, input_width + 2 * padding + stride - 1))
    for h in range(filter_size):
        height_max = h + stride * output_height
        for w in range(filter_size):
            width_max = w + stride * output_width
            img[:, :, h:height_max:stride, w:width_max:stride] += input_col[:, :, h, w, :, :]
    return img[:, :, padding:input_height + padding, padding:input_width + padding]

def forward_with_in2col(input:np.ndarray, weights , biases,stride=1, padding=0 ):
    batch_size, channel, input_height, input_width = input.shape
    count_filters, _ , filter_size, filter_size = weights.shape
    output_height = (input_height + 2 * padding - filter_size) // stride + 1
    output_width = (input_width + 2 * padding - filter_size) // stride + 1
    input_col = im2col(input, filter_size, stride, padding)
    weight_col = weights.reshape(count_filters, channel * filter_size * filter_size)
    output = np.dot(weight_col, input_col) + biases
    output = output.reshape(-1, batch_size, output_height, output_width).transpose(1, 0, 2, 3)
    return output

# 1. Циклический метод
def conv_loops(signal, kernels, bias):
    batch_size, _, height, width = signal.shape
    kernel_count, input_count, kernel_height, kernel_width = kernels.shape
    output_height = height - kernel_height + 1
    output_width = width - kernel_width + 1
    output_signal = np.zeros((batch_size, kernel_count, output_height, output_width))

    for b in range(batch_size):
        for c in range(input_count):
            for k in range(kernel_count):
                for i in range(output_height):
                    for j in range(output_width):
                        output_signal[b, k, i, j] += np.sum(
                            signal[b, c, i:i + kernel_height, j:j + kernel_width] * kernels[k, c, :, :]
                        ) + bias[k]
    return output_signal


def conv2d_with_loops(input_signal, kernels, biases, stride=1, padding=0):
    batch_size, channels, input_height, input_width = input_signal.shape
    num_filters, _ , kernel_height, kernel_width = kernels.shape

    # Определение размеров выходного изображения после свёртки
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    # Инициализация выходного тензора
    output = np.zeros((batch_size, num_filters, output_height, output_width))

    # Паддинг (добавляем нули по краям)
    if padding > 0:
        input_signal = np.pad(input_signal,
                              ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                              mode='constant')

    # Свёртка через циклы
    for b in range(batch_size):  # Перебор по batch
        for f in range(num_filters):  # Перебор по фильтрам
            for i in range(0, output_height):  # Перебор по высоте выходного тензора
                for j in range(0, output_width):  # Перебор по ширине выходного тензора
                    # Определяем окно для текущей позиции
                    for c in range(channels):  # Перебор по каналам
                        for m in range(kernel_height):  # Перебор по высоте ядра
                            for n in range(kernel_width):  # Перебор по ширине ядра
                                # Позиция в выходном тензоре
                                current_height = i * stride + m
                                current_width = j * stride + n
                                output[b, f, i, j] += (
                                        input_signal[b, c, current_height, current_width] *
                                        kernels[f, c, m, n]
                                )
                    # Добавляем смещение (bias)
                    output[b, f, i, j] += biases[f]

    return output

def einsum_forward(signal, kernels, biases):
    kernel_size = kernels.shape[2:3]
    # if signal[0].shape != self.input_size:
    #     raise Exception(f"Размерность сигнала {signal[0].shape} отличается от " +
    #                     f"размерности входящего слоя: {self.input_size}")

    wind = np.lib.stride_tricks.sliding_window_view(signal, kernel_size, axis=(2, 3))
    summator = np.einsum("bcijhw,ckhw->bkij", wind, kernels) + biases
    return summator #activation_function.calc(self.summator)

def test_forward():
    input_s = np.array(rng.random((8, 3, 7, 7)))
    # input_s = np.array([[
    #     [[1, 2, 3, 0],
    #      [4, 5, 6, 0],
    #      [7, 8, 9, 0],
    #      [0, 0, 0, 0]],
    #     [[1, 2, 3, 0],
    #      [4, 5, 6, 0],
    #      [7, 8, 9, 0],
    #      [0, 0, 0, 0]]
    # ]])# shape: (1, 2, 4, 4)
    kernels = np.array(rng.random((5, 3, 3, 3)))
    # kernels = np.array([[
    #     [[1, 0],
    #      [0, -1]],
    #     [[1, 0],
    #      [0, -1]]
    # ]])  # shape: (1, 2, 2, 2)
    biases = np.array(rng.random((5, 1)))
    # biases = np.array([0])  # shape: (1,)
    expected_output = np.array([[
        [[-4, -4, 3],
         [-4, -4, 6],
         [7, 8, 9]]
    ]])  # shape: (1, 1, 3, 3)

    loops = conv_loops(input_s, kernels, biases)
    print(f"loops: {loops}")
    im2c = forward_with_in2col(input_s, kernels, biases)
    print(f"im2c: {im2c}")
    print(f"shapes: loops={loops.shape}, im2c={im2c.shape}")
    print(f"{np.isclose(loops, im2c, rtol=1e-7)}")

def test_calc():
    lay = Convolution2D(1, (3, 3), (1, 4, 4), Functions.none)
    weight = np.array([[1, 4, 1],
                       [1, 4, 3],
                       [3, 3, 1]])
    lay.kernels = weight.reshape((1, 1, 3, 3))
    lay.kernels_col = weight.reshape((1, 1*3*3))
    lay.bias = np.zeros((1, 1))
    inp = np.array([[4, 5, 8, 7],
                    [1, 8, 8, 8],
                    [3, 6, 6, 4],
                    [6, 5, 7, 8]]).reshape((1, 1, 4, 4))
    res = lay.forward(inp)
    print(f"res: {res}")
    pred_delta = np.array([[2, 1],
                           [4, 4]]).reshape((1, 1, 2, 2))
    delta = lay.backward(pred_delta, is_need_out=True)
    delta_expextec = np.array([[2,  9,  6,  1],
                               [6, 29, 30,  7],
                               [10, 29, 33, 13],
                               [12, 24, 16,  4]])
    print(f"delta: {delta}")
    print(f"Похоже ли на ожидаемое: {np.isclose(delta.reshape((4,4)), delta_expextec)}")

def test_rot():
    weight = rng.random((2,2,3,3))
    kernels_col = weight.reshape((2, 2*3*3))
    kernels_col90 = np.rot90(kernels_col, 2)
    weight90 = np.rot90(weight, 2, axes=(2,3))
    kernels90_col = weight90.reshape((2, 2*3*3))
    print(f"\nkernels: {weight}")
    print(f"im2col->90: {kernels_col90}")
    print(f"kernels90:  {weight90}")
    print(f"90->im2col: {kernels90_col}")
    print(f"without 90: {kernels_col}")

    print(f"my_rot:     {my_rot180(kernels_col, 2, 3)}")

def my_rot180(im2col: np.ndarray, channels: int, filter_size:int):
    res=np.zeros_like(im2col)
    for i in range(channels):
        p =  np.fliplr(im2col[:,i*filter_size**2:(i+1)*filter_size**2])
        res[:,i*filter_size**2:(i+1)*filter_size**2] = np.fliplr(im2col[:,i*filter_size**2:(i+1)*filter_size**2])
    return res





