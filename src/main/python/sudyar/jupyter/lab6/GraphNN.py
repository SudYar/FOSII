from tqdm import tqdm
from src.main.python.sudyar.jupyter.lab6.Node import Node
from src.main.python.sudyar.jupyter.lab6.Functions import Functions, rng, dmae, mae, initialize_weights
import src.main.python.sudyar.jupyter.lab6.autograd_ops as autograd
import numpy as np

from src.main.python.sudyar.jupyter.lab6.Dense import Dense
from src.main.python.sudyar.jupyter.lab6.Flatten import Flatten
from src.main.python.sudyar.jupyter.lab6.MegNetBlock import MegNetBlock


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[...,0]
    get_y = lambda z: z[...,1]
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])

class GraphNN:
    def __init__(self):
        self.layers = [
            MegNetBlock(48, 24, megnet_output_sizes=(12, 24, 6), lay_name=f'M1'),
            MegNetBlock(24, 12, megnet_output_sizes=(6, 12, 3), lay_name=f'M2'),
            Flatten(lay_name='F1'),
            Dense(units=74, activation=Functions.relu, lay_name='D1'),
            Dense(units=42, activation=Functions.relu, lay_name='D2'),
            Dense(units=1, activation=Functions.sigmoid, lay_name='D3')
        ]
        self.history = {
            'loss': [],
            'grad': {}
        }
        self.is_compiled = False
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
    @property
    def params(self):
        return {
            **{f"MegNet0_{key}": value for key, value in self.layers[0].get_params().items()},
            **{f"MegNet1_{key}": value for key, value in self.layers[1].get_params().items()},
            **{f"Dense0_{key}": value for key, value in self.layers[3].get_params().items()},
            **{f"Dense1_{key}": value for key, value in self.layers[4].get_params().items()},
            **{f"Dense2_{key}": value for key, value in self.layers[5].get_params().items()}
        }

    def compile(self, input_shapes):
        self.is_compiled = True
        units_in = input_shapes
        for layer in self.layers:
            units_in = layer.compile(units_in)

        self.t = 0
        self.m = {name: np.zeros_like(param) for name, param in self.params.items()}
        self.v = {name: np.zeros_like(param) for name, param in self.params.items()}

    def _feedforward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output


    def _backprop(self, dloss):
        last = self.layers[-1].output
        last.ref_count = 1
        last.backward(dloss)


    def _update_params(self, learning_rate: float):
        self.t += 1
        for name, param in self.params.items():
            dparam = param.reset_grad()  # градиент для конкретного параметра

            key = name

            if key in self.history['grad']:
                self.history['grad'][key].append(dparam)
            else:
                self.history['grad'][key] = [dparam]

            # Обновляем моменты
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * dparam
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (dparam ** 2)

            # Коррекция на смещение
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Обновляем параметры
            param.value -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


    def fit(self, X:tuple[np.ndarray], y:np.ndarray, epoch_count:int=50, batch_size=16, learning_rate:float=1e-3, need_print=True ):
        nodes, edges, ues, connects = X
        if not self.is_compiled:
            self.compile((nodes.shape, edges.shape, ues.shape))
        total_samples = nodes.shape[0]

        # optimizer = optimizer(self.layers, lr=learning_rate)

        dataset = list(zip(X, y))

        iterator = range(epoch_count)
        if need_print: iterator = tqdm(iterator)

        for epoch in iterator:
            # Перемешиваем индексы
            indices = np.arange(total_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0
            # Проходимся по батчам
            for start_idx in range(0, total_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_idx = indices[start_idx:end_idx]
                # Формируем мини-батч
                X_batch = [Node(nodes[batch_idx]), Node(edges[batch_idx]), Node(ues[batch_idx]), connects[batch_idx]]

                y_batch = y[batch_idx]  # (batch_size, ...)

                num_batches += 1
                last = self._feedforward(X_batch)
                # print(i, self.output_layer.output_signal.max())

                res = last.value.reshape(-1)
                loss = mae(res, y_batch)
                grad_errors = dmae(res, y_batch)

                self._backprop(grad_errors.reshape(-1, 1))
                self._update_params(learning_rate)
                epoch_loss += loss

                if need_print: iterator.set_postfix({'Loss': loss})

            epoch_loss /= num_batches
            self.history['loss'].append(epoch_loss)


    def predict(self, X:np.ndarray):
        nodes, edges, ues, connects = X
        X = (Node(nodes), Node(edges), Node(ues), connects)
        return self._feedforward(X)
