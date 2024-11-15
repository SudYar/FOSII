import numpy as np

from Layer import Layer


class Maxpooling(Layer):

    def __init__(self, pool_size, strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size

        self.__input_shape = None
        self.mask = None


    def forward(self, signal: np.ndarray):
        self.__input_shape = signal.shape
        batch_size, channel, input_height, input_width = self.__input_shape
        output_height = (input_height - self.pool_size) // self.strides + 1
        output_width = (input_width - self.pool_size) // self.strides + 1
        windows = np.lib.stride_tricks.as_strided(signal,
                                                  shape=(
                                                      batch_size, channel, output_height, output_width, self.pool_size,
                                                      self.pool_size),
                                                  strides=(signal.strides[0], signal.strides[1],
                                                           self.strides * signal.strides[2],
                                                           self.strides * signal.strides[3],
                                                           signal.strides[2], signal.strides[3])
                                                  )
        output = np.max(windows, axis=(4, 5))

        maxs = output.repeat(2, axis=2).repeat(2, axis=3)
        x_window = signal[:, :, :output_height * self.strides, :output_width * self.strides]
        self.mask = np.equal(x_window, maxs).astype(int)
        return output

    def backward(self, error_signal: np.ndarray, is_need_out=True) -> np.ndarray | None:
        delta_conv = error_signal.repeat(self.pool_size, axis=2).repeat(self.pool_size, axis=3)
        delta_masked = np.multiply(delta_conv, self.mask)
        delta = np.zeros(self.__input_shape)
        delta[:, :, :delta_masked.shape[2], :delta_masked.shape[3]] = delta_masked
        return delta

    def update_params(self, learning_rate: float):
        pass