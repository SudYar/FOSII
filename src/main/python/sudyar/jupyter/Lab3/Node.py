import numpy as np


class Node:
    name:   str
    value:  np.ndarray
    grad:   np.ndarray | None
    parent: list['Node']
    ref_count: int

    def __repr__(self):
        return f"Node(name={self.name:<10}, " \
               f"value_shape={self.value.shape}, " \
               f"grad_shape={None if self.grad is None else self.grad.shape}, " \
               f"parents={[(parent.name, parent.ref_count) for parent in self.parents]})"

    def __init__(self, value: np.ndarray, parents: list['Node'] | None = None, grad_fn=None, name: str = "noname"):
        self.value = value
        self.grad = None  # Градиент, который будет накоплен
        self.parents = parents if parents else []
        self.grad_fn = grad_fn  # Функция для backward
        self.ref_count = 0  # Сколько потомков ссылаются на этот узел
        self.name = name

    def compute_grads(self, grad_output: np.ndarray):
        if self.grad_fn:
            return self.grad_fn(grad_output)
        return []

    def backward(self, grad_output: np.ndarray, need_print=False):
        """
        Выполняет обратный проход.
        grad_output: градиент, поступающий от следующего узла.
        """
        # Если grad_output — скаляр, преобразуем его в массив
        if isinstance(grad_output, (int, float)):
            grad_output = np.array(grad_output)

        # Проверяем размерности перед сложением
        if self.value.shape != grad_output.shape:
            raise ValueError(
                f"Shape mismatch in backward pass: self.grad.shape={self.grad.shape}, "
                f"grad_output.shape={grad_output.shape}"
            )
        if self.grad is None:
            self.grad = grad_output
        else:
            self.grad += grad_output
        # Уменьшаем счетчик ссылок
        self.ref_count -= 1
        # Если все потомки обработаны, передаём градиенты родителям
        if self.ref_count == 0:
            if need_print:
                print(self.__repr__())
            # Вычисляем градиенты для родителей
            for parent, grad in zip(self.parents, self.compute_grads(self.grad)):
                parent.backward(grad, need_print)

    def reset_grad(self) -> np.ndarray:
        grad, self.grad = self.grad, None
        return grad

import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph_with_networkx(in_nodes, scale=1.5, node_size=2000, font_size=10):
    def _collect_nodes(node, nodes):
        """Собирает всех родителей узла и добавляет их в nodes, если их там ещё нет"""
        if node not in nodes:
            nodes.append(node)
        for parent in node.parents:
            _collect_nodes(parent, nodes)
        return nodes

    # Создание графа
    G = nx.DiGraph()
    nodes = in_nodes.copy()
    for node in in_nodes:
        _collect_nodes(node, nodes)
    for i, node in enumerate(nodes):
        G.add_node(i, label=f"{node.name} ({i})\n\t({node.ref_count})")
        for parent in node.parents:
            G.add_edge(nodes.index(parent), i)

    # Определяем расположение узлов с увеличенным расстоянием между ними
    pos = nx.spring_layout(G, k=scale / len(nodes))  # Масштабируем расстояние

    # Настраиваем размер рисунка
    plt.figure(figsize=(12, 8))

    # Рисуем граф
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=nx.get_node_attributes(G, 'label'),
        node_size=node_size,
        font_size=font_size
    )
    plt.show()

def test_visual():
    node1 = Node(np.array([1, 2]), name="Input")
    node2 = Node(np.array([3, 4]), parents=[node1], name="Hidden")
    node3 = Node(np.array([5, 6]), parents=[node2], name="Output")

    # Список начальных узлов
    in_nodes = [node3]
    visualize_graph_with_networkx(in_nodes)
