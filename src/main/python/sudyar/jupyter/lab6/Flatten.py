import src.main.python.sudyar.jupyter.lab6.autograd_ops as autograd
import numpy as np
from src.main.python.sudyar.jupyter.lab6.Node import Node


class Flatten:
    def __init__(self, lay_name:str=''):
        self.lay_name=lay_name
        pass
    def compile(self, input_shapes):
        node_shape, edge_shape, u_shape = input_shapes
        f = [node_shape[-1], edge_shape[-1], u_shape[-1]]
        n = [node_shape[-2], edge_shape[-2]]
        self.output_dim = f[0] * n[0] + f[1] * n[1] + f[2]
        return self.output_dim

    def forward(self, inputs:tuple):
        nodes, edges, ues, connect = inputs
        batch = nodes.value.shape[0]
        self.shape = (batch, -1)
        nodes_flat = autograd.reshape(nodes, self.shape, name=f'{self.lay_name+"_"}nodes_flat')
        edges_flat = autograd.reshape(edges, self.shape, name=f'{self.lay_name+"_"}edges_flat')
        ues_flat   = autograd.reshape(ues, self.shape, name=f'{self.lay_name+"_"}ues_flat')

        self.out = autograd.concat([nodes_flat, edges_flat, ues_flat], axis=-1, name=f'{self.lay_name+"_"}conc')
        return self.out

    def get_params(self):
        return {}
