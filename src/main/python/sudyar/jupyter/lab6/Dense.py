from src.main.python.sudyar.jupyter.lab6.Node import Node, Parameter
from src.main.python.sudyar.jupyter.lab6.Functions import Functions, rng, dmae, mae, initialize_weights
import src.main.python.sudyar.jupyter.lab6.autograd_ops as autograd
import numpy as np
class Dense:
    def __init__(self, units: int, activation:Functions=Functions.none, lay_name:str=''):
        """
        Initialize the Dense layer.
        """
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.lay_name=lay_name

    @property
    def params(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        }

    def compile(self, in_shape: int | tuple):
        self.units_in = in_shape if isinstance(in_shape, int) else in_shape[-1]
        self.weights = Parameter(initialize_weights(in_size=self.units_in,
                                      size=(self.units_in, self.units),
                                      out_size=self.units, method='ksa'), name=f'{self.lay_name+"_"}W')
        self.bias = Parameter(np.array(rng.random(self.units) * 2 - 1).reshape(1, -1), name=f'{self.lay_name+"_"}b')
        return self.units

    def forward(self, input: Node):
        value = np.matmul(input.value, self.weights.value)
        value += self.bias.value

        self.output = autograd.linear_act(self.activation, input, self.weights, self.bias, name=f'{self.lay_name+"_"}linear_act')
        return self.output

    def get_params(self):
        return self.params

    def update_params(self, learning_rate: float):
        for param_name, param_node in self.params.items():
            param_node.value -= learning_rate * param_node.reset_grad()


def test_init():
    test = Dense(5)
    test.compile(10)
    res = test.forward(Node(rng.random(size=(1, 10))))
    res.ref_count = 1
    res.backward(rng.random(size=(1, 5)))
    print(test.params)