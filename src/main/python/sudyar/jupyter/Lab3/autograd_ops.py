import itertools
from typing import Optional

import numpy as np
from Node import Node
from src.main.python.sudyar.jupyter.GraphMath.Functions import Functions


def create_node(value, parents, grad_fn, name="noname"):
    # for parent in parents:
    #     parent.ref_count += 1
    [setattr(parent, 'ref_count', parent.ref_count + 1) for parent in parents]
    return Node(value, parents=parents, grad_fn=grad_fn, name=name)


def matmul(a: Node, b: Node, name='matmul') -> Node:
    value = np.matmul(a.value, b.value)

    def grad_fn(grad_output):
        grad_a = np.matmul(grad_output, b.value.T)
        grad_b = np.matmul(a.value.T, grad_output)
        return [grad_a, grad_b]

    return create_node(value, parents=[a, b], grad_fn=grad_fn, name=name)


def multiply(a: Node, b:Node, name='multiply') -> Node:
    assert a.value.shape == b.value.shape
    value = a.value * b.value

    def grad_fn(grad_output):
        grad_a = grad_output * b.value
        grad_b = a.value * grad_output
        return [grad_a, grad_b]

    return create_node(value, parents=[a, b], grad_fn=grad_fn, name=name)


def multiply_with_slice(a: Node, b: Node, slice_indices: tuple[Optional[int], Optional[int]], name='multiply_with_slice') -> Node:
    """
    Умножает часть одного узла на второй узел.
    :param a: Узел, который будет разрезан (например, past_node).
    :param b: Узел, с которым нужно умножить часть первого узла.
    :param slice_indices: Кортеж (start, end) для индексов, определяющих часть a.value.
    :param name: Имя для нового узла.
    """
    start, end = slice_indices
    # slicing = create_slice(dim_count=len(a.value.shape), axis=axis, start=start, end=end)
    sliced_a_value = a.value[:, start:end]
    assert sliced_a_value.shape == b.value.shape, "Shapes must match for multiplication."
    value = a.value.copy()
    value[:, start:end] = sliced_a_value * b.value

    def grad_fn(grad_output):

        grad_b = np.multiply(grad_output[:, start:end], a.value[:, start:end])  # Градиент для b
        grad_a = grad_output  # Градиент для a ЭТО НЕ COPY, ТАК ЧТО ОСТОРОЖНО
        grad_a[:, start:end] = grad_output[:, start:end] * b.value  # Только в срезе


        return [grad_a, grad_b]

    return create_node(value, parents=[a, b], grad_fn=grad_fn, name=name)


def add(a: Node, b: Node, name='add') -> Node:
    value = a.value + b.value

    def grad_fn(grad_output):
        grad_a = grad_output
        grad_b = np.sum(grad_output, axis=0, keepdims=True) if b.value.shape[0] < grad_output.shape[0] else grad_output
        return [grad_a, grad_b]

    return create_node(value, parents=[a, b], grad_fn=grad_fn, name=name)


def minus(a: Node|np.ndarray, b: Node|np.ndarray, name='minus') -> Node:
    val_a, val_b = a, b
    parents = []
    if isinstance(a, Node):
        parents.append(a)
        val_a = a.value
    if isinstance(b, Node):
        parents.append(b)
        val_b = b.value
    value = np.subtract(val_a, val_b)

    def grad_fn(grad_output):
        grads = []
        if isinstance(a, Node): grads.append(grad_output)
        if isinstance(b, Node): grads.append(-grad_output)
        return grads

    return create_node(value, parents=parents, grad_fn=grad_fn, name=name)


def act(f: Functions, a: Node, name='act'):
    value = f.calc(a.value)

    def grad_fn(grad_output):
        grad_a = grad_output * f.derivative(value)
        return [grad_a]

    return create_node(value, parents=[a], grad_fn=grad_fn, name=name)


def linear(x:Node, w:Node, b:Node|None=None, name='linear'):
    value = np.matmul(x.value, w.value) + b.value if b else 0
    nodes = [x, w]
    if b:
        nodes.append(b)

    def grad_fn(grad_output):
        grad_x = np.matmul(grad_output, w.value.T)
        grad_w = np.matmul(x.value.T, grad_output)
        grads = [grad_x, grad_w]
        if b:
            grad_b = np.sum(grad_output, axis=0, keepdims=True) if b.value.shape[0] < grad_output.shape[
                0] else grad_output
            grads.append(grad_b)
        return grads
    return create_node(value, parents=nodes, grad_fn=grad_fn, name=name)


def linear_act(f: Functions, x:Node, w:Node, b:Node|None=None, name='linear_act'):
    value = np.matmul(x.value, w.value) \
            + b.value if b else 0
    value = f.calc(value)
    nodes = [x, w]
    if b:
        nodes.append(b)

    def grad_fn(grad_output):
        grad = grad_output * f.derivative(value)
        grad_x = np.matmul(grad, w.value.T)
        grad_w = np.matmul(x.value.T, grad)
        grads = [grad_x, grad_w]
        if b:
            grad_b = np.sum(grad, axis=0, keepdims=True) if b.value.shape[0] < grad_output.shape[
                0] else grad
            grads.append(grad_b)
        return grads

    return create_node(value, parents=nodes, grad_fn=grad_fn, name=name)

def concat(nodes: list[Node], axis: int=-1, name='concat'):
    values = [node.value for node in nodes]  # Извлекаем значения из узлов
    concatenated_value = np.concatenate(values, axis=axis)  # Выполняем объединение значений

    def grad_fn(grad_output):
        grads = []
        start_idx = 0
        for node in nodes:
            # Определяем, какой кусок градиента относится к текущему узлу
            size = node.value.shape[axis]
            slice_indices = [slice(None)] * grad_output.ndim
            slice_indices[axis] = slice(start_idx, start_idx + size)
            grads.append(grad_output[tuple(slice_indices)])
            start_idx += size
        return grads

    return create_node(concatenated_value, parents=nodes, grad_fn=grad_fn, name=name)

def model_forw_with_concat(input_signal_node1, past_node1, update_gate_weights1, bias_z1,
                          reset_gate_weights1, bias_r1, candidate_weights1, bias_h1):
    concat_node1 = concat([input_signal_node1, past_node1])
    z_gate1 = linear_act(Functions.sigmoid, concat_node1, update_gate_weights1, bias_z1, name='z_gate(act)')
    r_gate1 = linear_act(Functions.sigmoid, concat_node1, reset_gate_weights1, bias_r1, name='r_gate(act)')

    reset_info = multiply(past_node1, r_gate1)
    reset_inp1 = concat([input_signal_node1, reset_info])

    h_hat_act1 = linear_act(Functions.tanh, reset_inp1, candidate_weights1, bias_h1,
                            name='h_hat(act)')
    update_gate1 = minus(np.ones_like(z_gate1.value), z_gate1)
    updated_past1 = multiply(update_gate1, past_node1, name='i_old')
    updated_new1 = multiply(z_gate1, h_hat_act1, name='i_new')
    result1 = add(updated_past1, updated_new1)
    return result1

def model_forw_with_slice(input_signal_node2, past_node2, update_gate_weights2, bias_z2,
                          reset_gate_weights2, bias_r2, candidate_weights2, bias_h2):

    concat_node2 = concat([input_signal_node2, past_node2])
    z_gate2 = linear_act(Functions.sigmoid, concat_node2, update_gate_weights2, bias_z2, name='z_gate(act)')
    r_gate2 = linear_act(Functions.sigmoid, concat_node2, reset_gate_weights2, bias_r2, name='r_gate(act)')

    reset_inp2 = multiply_with_slice(concat_node2, r_gate2, (input_signal_node2.value.shape[-1], None))

    h_hat_act2 = linear_act(Functions.tanh, reset_inp2, candidate_weights2, bias_h2,
                            name='h_hat(act)')
    update_gate2 = minus(np.ones_like(z_gate2.value), z_gate2)
    updated_past2 = multiply(update_gate2, past_node2, name='i_old')
    updated_new2 = multiply(z_gate2, h_hat_act2, name='i_new')
    result2 = add(updated_past2, updated_new2)
    return  result2


def test_simple():
    batch_size = 4
    seq_len = 10
    input_size = 16
    hidden_size = 8
    import numpy as np

    np.random.seed(42)  # Для повторяемости

    # Случайные данные
    input_signal = np.random.randn(batch_size, input_size)
    past_node_value = np.random.randn(batch_size, hidden_size)
    r_gate_value = np.random.uniform(0, 1, (batch_size, hidden_size))

    input_signal_node1 = Node(input_signal, name="input_signal")
    past_node1 = Node(past_node_value, name="past_node")
    r_gate1 = Node(r_gate_value, name="r_gate")

    input_signal_node2 = Node(input_signal, name="input_signal")
    past_node2 = Node(past_node_value, name="past_node")
    r_gate2 = Node(r_gate_value, name="r_gate")

    reset_info = multiply(past_node1, r_gate1)
    reset_inp_1 = concat([input_signal_node1, reset_info])

    concat_node = concat([input_signal_node2, past_node2])
    reset_inp_2 = multiply_with_slice(concat_node, r_gate2, (input_size, None))

    print("Are reset_inp values identical?")
    print(np.allclose(reset_inp_1.value, reset_inp_2.value))

    grad_output = np.random.randn(*reset_inp_1.value.shape)
    reset_inp_1.ref_count = 1
    reset_inp_2.ref_count = 1
    reset_inp_1.backward(grad_output, need_print=True)
    print('next\n\n')
    reset_inp_2.backward(grad_output, need_print=True)

    # Вычислить градиенты для первого подхода
    grads_1 = [input_signal_node1.grad, r_gate1.grad, past_node1.grad]

    # Вычислить градиенты для второго подхода
    grads_2 = [input_signal_node2.grad, r_gate2.grad, past_node2.grad]

    etalot_grad_r_gate = grad_output[:, input_size:] * past_node_value

    # Сравнить градиенты
    for i, (g1, g2) in enumerate(zip(grads_1, grads_2)):
        print(f"Are gradients for input {i} identical?")
        print(np.allclose(g1, g2))

    print(np.allclose(r_gate1.grad,etalot_grad_r_gate))
    print(np.allclose(r_gate2.grad,etalot_grad_r_gate))


def test_concat_and_slice():
    import numpy as np
    batch_size = 4
    seq_len = 10
    input_size = 16
    hidden_size = 16

    np.random.seed(42)  # Для повторяемости

    # Случайные данные
    input_signal = np.random.randn(batch_size, input_size)
    past_node_value = np.random.randn(batch_size, hidden_size)
    # r_gate_value = np.random.uniform(0, 1, (batch_size, hidden_size))
    update_gate_weights_value = np.random.randn(input_size+hidden_size, hidden_size)
    reset_gate_weights_value = np.random.randn(input_size+hidden_size, hidden_size)
    candidate_weights_value = np.random.randn(input_size+hidden_size, hidden_size)
    bias_z_value = np.random.randn(1, hidden_size)
    bias_r_value = np.random.randn(1, hidden_size)
    bias_h_value = np.random.randn(1, hidden_size)


    input_signal_nodeT1 = Node(input_signal, name="input_signal")
    input_signal_nodeT2 = Node(input_signal, name="input_signal")
    past_node1 = Node(past_node_value, name="past_node")
    update_gate_weights1 = Node(update_gate_weights_value, name="upd_w_node")
    bias_z1 = Node(bias_z_value, name='upd_b_node')
    reset_gate_weights1 = Node(reset_gate_weights_value, name="upd_w_node")
    bias_r1 = Node(bias_r_value, name='upd_b_node')
    candidate_weights1 = Node(candidate_weights_value, name="upd_w_node")
    bias_h1 = Node(bias_h_value, name='upd_b_node')

    resultT1 = model_forw_with_concat(input_signal_nodeT1, past_node1, update_gate_weights1, bias_z1,
                          reset_gate_weights1, bias_r1, candidate_weights1, bias_h1)
    resultT2 = model_forw_with_concat(input_signal_nodeT2, resultT1, update_gate_weights1, bias_z1,
                          reset_gate_weights1, bias_r1, candidate_weights1, bias_h1)
    # resultT1_2 = model_forw_with_concat(resultT1, Node(past_node_value, name="past_node"), update_gate_weights1, bias_z1,
    #                       reset_gate_weights1, bias_r1, candidate_weights1, bias_h1)
    # resultT2_2 = model_forw_with_concat(resultT2, resultT1_2, update_gate_weights1, bias_z1,
    #                                     reset_gate_weights1, bias_r1, candidate_weights1, bias_h1)


    input_signal_node2T1 = Node(input_signal, name="input_signal")
    input_signal_node2T2 = Node(input_signal, name="input_signal")
    past_node2 = Node(past_node_value, name="past_node")
    update_gate_weights2 = Node(update_gate_weights_value, name="upd_w_node")
    bias_z2 = Node(bias_z_value, name='upd_b_node')
    reset_gate_weights2 = Node(reset_gate_weights_value, name="upd_w_node")
    bias_r2 = Node(bias_r_value, name='upd_b_node')
    candidate_weights2 = Node(candidate_weights_value, name="upd_w_node")
    bias_h2 = Node(bias_h_value, name='upd_b_node')

    result2T1 = model_forw_with_slice(input_signal_node2T1, past_node2, update_gate_weights2, bias_z2,
                          reset_gate_weights2, bias_r2, candidate_weights2, bias_h2)
    result2T2 = model_forw_with_slice(input_signal_node2T2, result2T1, update_gate_weights2, bias_z2,
                          reset_gate_weights2, bias_r2, candidate_weights2, bias_h2)
    # result2T1_2 = model_forw_with_slice(result2T1, Node(past_node_value, name="past_node"), update_gate_weights2, bias_z2,
    #                       reset_gate_weights2, bias_r2, candidate_weights2, bias_h2)
    # result2T2_2 = model_forw_with_slice(result2T2, result2T1_2, update_gate_weights2, bias_z2,
    #                                     reset_gate_weights2, bias_r2, candidate_weights2, bias_h2)





    print("Are reset_inp values identical?")
    print(np.allclose(resultT2.value, result2T2.value))

    grad_output = np.random.randn(*resultT2.value.shape)
    resultT1.grad = grad_output
    result2T1.grad = grad_output

    resultT2.ref_count = 1
    result2T2.ref_count = 1
    resultT2.backward(grad_output)
    result2T2.backward(grad_output)
    # Вычислить градиенты для первого подхода
    grads_1 = [input_signal_nodeT1.grad, past_node1.grad, reset_gate_weights1.grad]

    # Вычислить градиенты для второго подхода
    grads_2 = [input_signal_node2T1.grad, past_node2.grad, reset_gate_weights2.grad]

    # Сравнить градиенты
    for i, (g1, g2) in enumerate(zip(grads_1, grads_2)):
        print(f"Are gradients for input {i} identical?")
        print(np.allclose(g1, g2))



