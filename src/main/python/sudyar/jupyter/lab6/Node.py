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
               f'{"" if self.grad is None else f"grad_min={self.grad.min():.4f}, grad_max={self.grad.max():.4f}, grad_mean={self.grad.mean():.4f}, "}' \
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

        if self.grad is None:
            self.grad = grad_output.copy()
        else:
            # Проверяем размерности перед сложением
            if self.value.shape != grad_output.shape:
                raise ValueError(
                    f"Shape mismatch in backward pass: self.grad.shape={self.grad.shape}, "
                    f"grad_output.shape={grad_output.shape}"
                )
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


class Parameter(Node):

    def __repr__(self):
        return f"Node(name={self.name:<10}, " \
               f"value_shape={self.value.shape}, " \
               f"_min={self.value.min():.4f}, _max={self.value.max():.4f}, _mean={self.value.mean():.4f}" \
               f'\n{"" if self.grad is None else f",grad_shape={self.grad.shape}"}' \
               f'{"" if self.grad is None else f", grad_min={self.grad.min():.6f}, grad_max={self.grad.max():.6f}, grad_mean={self.grad.mean():.6f}"}'



import plotly.graph_objects as go


def visualize_computation_graph_with_plotly(root_node):
    """
    Визуализация графа вычислений с использованием plotly и выделенных окончаний рёбер.

    :param root_node: Узел, являющийся начальной точкой графа.
    """
    nodes = []
    edges = []
    labels = []
    positions = {}
    global_pos = [0]  # Глобальный счетчик позиции для узлов на уровне

    def traverse(node, level=0):
        """
        Рекурсивно проходит по графу и собирает информацию об узлах и связях.
        :param node: Текущий узел
        :param level: Глубина узла в графе
        """
        node_id = id(node)

        # Добавляем узел, если он еще не обработан
        if node_id not in positions:
            positions[node_id] = (global_pos[level], level)
            labels.append(
                f"{node.name}({node.value.shape})<br>"
                f"{'' if node.grad is None else f'grad_min={node.grad.min():.4f}, grad_max={node.grad.max():.4f}, grad_mean={node.grad.mean():.4f}<br>'}"
            )
            nodes.append(node)
            global_pos[level] += 1  # Увеличиваем позицию для следующего узла на этом уровне

            # Обрабатываем родителей
            if len(global_pos) <= level + 1:
                global_pos.append(np.random.random())  # Добавляем новый уровень
            for parent in node.parents:
                edges.append((id(parent), node_id))
                traverse(parent, level + 1)

    # Проходим весь граф
    traverse(root_node)

    # Координаты для отображения узлов
    x_nodes = [positions[id(node)][0] for node in nodes]
    y_nodes = [-positions[id(node)][1] for node in nodes]  # Инвертируем уровень для визуализации сверху вниз

    # Координаты для связей и наконечников
    edges_x, edges_y = [], []
    arrow_x, arrow_y = [], []

    for start, end in edges:
        x0, y0 = positions[start]
        x1, y1 = positions[end]

        # Линия между узлами
        edges_x += [x0, x1, None]
        edges_y += [-y0, -y1, None]

        # Наконечник стрелки
        dx, dy = x1 - x0, y1 - y0
        #norm = np.sqrt(dx ** 2 + dy ** 2)
        arrow_size = 0.3  # Размер наконечника
        arrow_width = 0.08  # Ширина наконечника
       # if norm != 0:  # Защита от деления на ноль
        ux, uy = dx, dy # / norm, dy / norm
        # Левое "крыло" стрелки
        left_x = x1 - ux * arrow_size #- uy * arrow_width
        left_y = y1 - uy * arrow_size #- ux * arrow_width
            # Правое "крыло" стрелки
            #right_x = x1 - ux * arrow_size + uy * arrow_width
            #right_y = -y1 + uy * arrow_size + ux * arrow_width
            # Добавляем точки наконечника
        arrow_x += [left_x, x1, None]
        arrow_y += [-left_y, -y1, None]

    # Создание графа с plotly
    fig = go.FigureWidget()

    # Добавляем линии (рёбра графа)
    fig.add_trace(go.Scatter(
        x=edges_x,
        y=edges_y,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))

    # Добавляем наконечники стрелок
    fig.add_trace(go.Scatter(
        x=arrow_x,
        y=arrow_y,
        mode='lines',
        line=dict(color='red', width=1),
        hoverinfo='none'
    ))

    # Добавляем узлы
    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        marker=dict(
            size=20,
            color='LightSkyBlue',
            line=dict(color='black', width=2)
        ),
        text=labels,
        textposition="middle left",
        textfont=dict(
            color='black',
            size=12
        ),
        hoverinfo='text'
    ))

    # Настройки отображения
    fig.update_layout(
        title="Граф вычислений с корректными стрелками",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.show()

def test_visual():
    node1 = Node(np.array([1, 2]), name="Input")
    node2 = Node(np.array([3, 4]), parents=[node1], name="Hidden")
    node3 = Node(np.array([5, 6]), parents=[node2], name="Output")

    # Корневой узел со всеми родителями
    in_nodes = node3
    visualize_computation_graph_with_plotly(in_nodes)
