from src.main.python.sudyar.jupyter.lab6.Node import Node
from src.main.python.sudyar.jupyter.lab6.Functions import Functions, rng, dmae, mae, initialize_weights
import src.main.python.sudyar.jupyter.lab6.autograd_ops as autograd
import numpy as np

from src.main.python.sudyar.jupyter.lab6.Dense import Dense


class MegNetBlock:
    def __init__(self, dd1:int=64, dd2:int=32, megnet_output_sizes:tuple=None, lay_name:str=''):
        self.lay_name=lay_name
        self.dd1 = dd1
        self.dd2 = dd2
        self.dense_node = [
            Dense(units=dd1, activation=Functions.relu, lay_name=f'{self.lay_name}_n1'),
            Dense(units=dd2, activation=Functions.relu, lay_name=f'{self.lay_name}_n2'),
        ]
        self.dense_edge = [
            Dense(units=dd1, activation=Functions.relu, lay_name=f'{self.lay_name}_e1'),
            Dense(units=dd2, activation=Functions.relu, lay_name=f'{self.lay_name}_e2'),
        ]
        self.dense_u = [
            Dense(units=dd1, activation=Functions.relu, lay_name=f'{self.lay_name}_u1'),
            Dense(units=dd2, activation=Functions.relu, lay_name=f'{self.lay_name}_u2'),
        ]

        # Если выходные размеры отличаются от входящих - запоминаем
        self.node_out_size, self.edge_out_size, self.u_out_size = megnet_output_sizes if megnet_output_sizes is not None else (None, None, None)
        # self.megnet = MegNet(node_size=self.node_size, edge_size=self.edge_size, u_size=self.u_size)

    def compile(self, input_shapes:tuple):
        node_shape, edge_shape, u_shape = input_shapes
        batch, self.n_nodes, self.f_nodes = node_shape
        _, self.n_edges, self.f_edges = edge_shape
        _, self.f_u = u_shape

        if self.node_out_size is None:
            self.node_out_size, self.edge_out_size, self.u_out_size = self.f_nodes, self.f_edges, self.f_u
        node_out_shape = (batch, self.n_nodes, self.node_out_size)
        edge_out_shape = (batch, self.n_edges, self.edge_out_size)
        u_out_shape    = (batch, self.u_out_size)

        self.dense_node[0].compile(node_shape)
        self.dense_edge[0].compile(edge_shape)
        self.dense_u[0].compile(u_shape)

        self.dense_node[1].compile(self.dd1)
        self.dense_edge[1].compile(self.dd1)
        self.dense_u[1].compile(self.dd1)


        self.dense_edge_out = Dense(units=self.edge_out_size, activation=Functions.softplus, lay_name=f'{self.lay_name}_e3')
        self.dense_edge_out.compile(4*self.dd2)

        self.dense_node_out = Dense(units=self.node_out_size, activation=Functions.softplus, lay_name=f'{self.lay_name}_n3')
        self.dense_node_out.compile(2*self.dd2 + self.edge_out_size)

        self.dense_u_out = Dense(units=self.u_out_size, activation=Functions.softplus, lay_name=f'{self.lay_name}_u3')
        self.dense_u_out.compile(self.node_out_size+self.edge_out_size+self.dd2)

        return (node_out_shape, edge_out_shape, u_out_shape)

    @property
    def params(self):
        return {
            **{f"Node0_{key}": value for key, value in self.dense_node[0].get_params().items()},
            **{f"Node1_{key}": value for key, value in self.dense_node[1].get_params().items()},
            **{f"Edge0_{key}": value for key, value in self.dense_edge[0].get_params().items()},
            **{f"Edge1_{key}": value for key, value in self.dense_edge[1].get_params().items()},
            **{f"U0_{key}": value for key, value in self.dense_u[0].get_params().items()},
            **{f"U1_{key}": value for key, value in self.dense_u[1].get_params().items()},
            **{f"Node_out_{key}": value for key, value in self.dense_node_out.get_params().items()},
            **{f"Edge_out_{key}": value for key, value in self.dense_edge_out.get_params().items()},
            **{f"U_out_{key}": value for key, value in self.dense_u_out.get_params().items()}
        }

    def forward(self, inputs):
        nodes, edges, ues, connects = inputs
        batch_size = connects.shape[0]
        # Проходим два полносвязных слоя и получаем размерности (batch, n_, dd2)
        for layer in self.dense_node:
            nodes = layer.forward(nodes)
        for layer in self.dense_edge:
            edges = layer.forward(edges)
        for layer in self.dense_u:
            ues = layer.forward(ues)

        V_s = autograd.gather_nodes(nodes, connects[..., 0], batch_size=batch_size,  name=f'{self.lay_name+"_"}gather_v_s')
        V_r = autograd.gather_nodes(nodes, connects[..., 1], batch_size=batch_size,  name=f'{self.lay_name+"_"}gather_v_r')
        u_rep_e = autograd.repeat(ues, self.n_edges, axis=1, name=f'{self.lay_name+"_"}u_rep_e')

        e_concat = autograd.concat([V_s, V_r, edges, u_rep_e], axis=-1, name=f'{self.lay_name+"_"}e_conc')
        self.edge_new = self.dense_edge_out.forward(e_concat)

        edge_mask = np.zeros(
            (batch_size, self.n_nodes, self.n_edges), dtype=bool
        )
        edge_mask[
            np.arange(batch_size)[:, None, None],
            connects[..., 1][:, None, :],
            np.arange(self.n_edges)
        ] = True

        aggregated_edges = autograd.einsum(self.edge_new, edge_mask.astype(int), 'bef', 'bie', 'bif', name=f'{self.lay_name+"_"}aggr_edges')
        # Усредняем по количеству рёбер на узел
        node_edge_counts = edge_mask.sum(axis=-1, keepdims=True)
        node_edge_counts[node_edge_counts == 0] = 1  # защита от деления на 0
        edge_mean = autograd.divide(aggregated_edges, node_edge_counts, name=f'{self.lay_name+"_"}div')
        u_rep_v = autograd.repeat(ues, self.n_nodes, axis=1, name=f'{self.lay_name+"_"}u_rep_v')

        v_concat = autograd.concat([edge_mean, nodes, u_rep_v], axis=-1, name=f'{self.lay_name+"_"}v_conc')
        self.node_new = self.dense_node_out.forward(v_concat)

        u_edge_mean = autograd.mean(self.edge_new, axis=1, name=f'{self.lay_name+"_"}u_edge_mean')  # (B, edge_size)
        u_node_mean = autograd.mean(self.node_new, axis=1, name=f'{self.lay_name+"_"}u_node_mean')

        u_concat = autograd.concat([u_edge_mean, u_node_mean, ues], axis=-1, name=f'{self.lay_name+"_"}u_conc')
        self.ues_new = self.dense_u_out.forward(u_concat)

        return (self.node_new, self.edge_new, self.ues_new, connects)

    def get_params(self):
        return self.params
