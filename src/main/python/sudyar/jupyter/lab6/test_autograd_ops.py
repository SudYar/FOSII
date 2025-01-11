import unittest
import numpy as np
import autograd_ops as autograd
from Node import Node


class TestAutoGrad(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_split(self):
        # Создаём исходный узел
        tensor_value = np.random.randn(4, 10, 8)  # [batch=4, seq_len=10, in_len=8]
        tensor_node = Node(tensor_value, name="input_tensor")

        # Разделяем тензор на узлы по временной оси (axis=1)
        split_nodes = autograd.split(tensor_node, axis=1)

        # Проверяем размерность и значения каждого узла
        for i, node in enumerate(split_nodes):
            print(f"Slice {i}: shape={node.value.shape}, value=\n{node.value}")

    def test_split_basic(self):
        # Создаем узел с тестовыми данными
        node = Node(np.array([[1, 2, 3], [4, 5, 6]]))  # shape (2, 3)

        # Выполняем разбиение с сохранением размерности
        result_nodes = autograd.split(node, axis=1, keepdims=True)
        # Проверяем количество полученных узлов
        self.assertEqual(len(result_nodes), 3)
        # Проверяем значения в каждом узле shape (2, 1)
        expected_values = [np.array([[1], [4]]), np.array([[2], [5]]), np.array([[3], [6]])]
        for expected_value, result_node in zip(expected_values, result_nodes):
            np.testing.assert_array_equal(result_node.value, expected_value)
        # Аналогично, но без сохранения
        result_nodes = autograd.split(node, axis=1, keepdims=False)

        self.assertEqual(len(result_nodes), 3)
        # Проверяем значения в каждом узле shape (2,)
        expected_values = [np.array([1, 4]), np.array([2, 5]), np.array([3, 6])]
        for expected_value, result_node in zip(expected_values, result_nodes):
            np.testing.assert_array_equal(result_node.value, expected_value)

    def test_split_grad_fn(self):
        # Создаем узел с тестовыми данными
        node = Node(np.array([[1, 2, 3], [4, 5, 6]]))  # shape (2, 3)

        # Выполняем разбиение
        result_nodes = autograd.split(node, axis=1, keepdims=True)
        # аналогично, но без сохранения размерности
        result_nodes2 = autograd.split(node, axis=1, keepdims=False)

        # Создаем градиентный выход для первого среза
        grad_output = np.array([[1], [-1]])
        grad_output2 = np.array([2, -2])

        # Вызываем функцию градиента для первого узла
        grads = result_nodes[0].grad_fn(grad_output)
        grads2 = result_nodes2[1].grad_fn(grad_output2)

        # Проверяем, что градиенты обновляются корректно
        expected_grad_tensor = np.zeros_like(node.value)
        expected_grad_tensor[0, 0] = 1  # Обновляем только соответствующий срез
        expected_grad_tensor[1, 0] = -1

        expected_grad_tensor2 = np.zeros_like(node.value)
        expected_grad_tensor2[0, 1] = 2  # Обновляем только соответствующий срез
        expected_grad_tensor2[1, 1] = -2

        np.testing.assert_array_equal(grads, expected_grad_tensor)
        np.testing.assert_array_equal(grads2, expected_grad_tensor2)


    def test_concat_basic(self):
        # Создаем узлы с тестовыми данными
        node1 = Node(np.array([[1, 2], [3, 4]]))
        node2 = Node(np.array([[5, 6], [7, 8]]))
        nodes = [node1, node2]

        # Выполняем конкатенацию
        result_node = autograd.concat(nodes, axis=0)
        result_node_nkeepdim = autograd.concat(nodes, axis=1, keepdim=False)

        # Проверяем результат
        expected_value = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_value2 = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]])
        np.testing.assert_array_equal(result_node.value, expected_value)
        np.testing.assert_array_equal(result_node_nkeepdim.value, expected_value2)

    def test_concat_grad_fn(self):
        # Создаем узлы с тестовыми данными
        node1 = Node(np.array([[1, 2], [3, 4]]))
        node2 = Node(np.array([[5, 6]]))
        nodes = [node1, node2]

        # Выполняем конкатенацию
        result_node = autograd.concat(nodes, axis=0)

        # Создаем градиентный выход
        grad_output = np.array([[6, 5], [4, 3], [2, 1]]).reshape(result_node.value.shape)

        # Получаем градиенты
        grads = result_node.grad_fn(grad_output)

        # Проверяем, что градиенты соответствуют ожидаемым
        expected_grads_node1 = np.array([[6, 5], [4, 3]])
        expected_grads_node2 = np.array([[2, 1]])

        np.testing.assert_array_equal(grads[0], expected_grads_node1)
        np.testing.assert_array_equal(grads[1], expected_grads_node2)

    def test_gather_nodes(self):
        # Данные для теста
        batch_size = 2
        node_dim = 3
        nodes_value = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        indices_value = np.array([[[0, 1]], [[1, 0]]])

        # Узлы
        nodes = Node(nodes_value)
        indices = indices_value

        # Выполнение операции
        gathered = autograd.gather_nodes(nodes, indices[..., 1], batch_size=batch_size)

        # Ожидаемые значения
        expected_value = np.array([[[4, 5, 6]], [[7, 8, 9]]]) # значения узлов 1 и 0 из 0 и 1 батчей соответственнр

        # Проверка прямого результата
        assert np.allclose(gathered.value, expected_value), "Gather nodes direct computation failed"

        # Проверка обратного распространения
        grad_output = np.ones_like(expected_value)
        grad = gathered.grad_fn(grad_output)
        expected_grad = np.array([[[0, 0, 0], [1, 1, 1]], [[1, 1, 1], [0, 0, 0]]]) # градиент возвращается только в конкретные узлы
        assert np.allclose(grad[0], expected_grad), "Gather nodes backward computation failed"

    def test_repeat(self):
        # Данные для теста
        input_value = np.array([[1, 2], [3, 4]])
        repeats = 3
        axis = 1

        # Узел
        input_node = Node(input_value)

        # Выполнение операции
        repeated = autograd.repeat(input_node, repeats, axis=axis)

        # Ожидаемые значения
        expected_value = np.array([[[1, 2], [1, 2],[1, 2]], [[3, 4], [3, 4], [3, 4]]])

        # Проверка прямого результата
        assert np.allclose(repeated.value, expected_value), "Repeat direct computation failed"

        # Проверка обратного распространения
        grad_output = np.ones_like(expected_value)
        grad = repeated.grad_fn(grad_output)
        expected_grad = np.ones_like(input_value) * repeats
        assert np.allclose(grad[0], expected_grad), "Repeat backward computation failed"


if __name__ == '__main__':
    unittest.main()
