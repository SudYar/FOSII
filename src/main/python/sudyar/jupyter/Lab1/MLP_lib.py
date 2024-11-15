import numpy as np
import pandas as pd
# Для тепловых карт
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 6)
rng = np.random.default_rng(51)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


sig_f = (sigmoid, dsigmoid)


def simpl_f(x):
    res = np.copy(x)
    res[res >= 0] = 1
    res[res < 0] = 0
    return res


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1 - np.square(tanh(x))


tanh_f = (tanh, dtanh)


# Средняя квадратичная ошибка
def mse(y_pred, y):
    # 1/n*sum((угаданное-Y)^2)
    err = np.mean(np.square(y_pred - y))
    return err


# y_pred - рассчитанное значение, y - ожидаемое
def dmse(y_pred, y):
    n = y.shape[1]
    return (2 / n) * (y_pred - y)


# Визуализация весов модели
def visualize_weights_with_biases(weights, biases, title='График'):
    """
  Делает тепловую карту для весов и смещения каждого нейрона каждого слоя

  -----
  :param weights: матрица размерности (k*i*j) где k- количество весов, i - нейроны текущего слоя, j - входы в нейрон
  :param biases: матрица размерности (k*i*1)
  :param title: Название для графиков (опционально)
  :return: none
  """
    for i, (weights, bias) in enumerate(zip(weights, biases)):
        # Транспонируем матрицу весов для замены строк и колонок
        weights_t = weights.T
        # добавляем смещение как дополнительную строку
        bias_row = bias.reshape(1, -1)
        combined_matrix = np.vstack([weights_t, bias_row])

        # Визуализируем тепловую карту
        plt.figure(figsize=(10, 6))
        sns.heatmap(combined_matrix, annot=True, cmap='coolwarm', cbar=False)
        plt.title(f'{title} для слоя {i + 1}')
        plt.xlabel('Нейроны')
        plt.ylabel('Вход и смещение(последнее)')
        plt.show()


def normalize(y_pred):
    """
  Нормализация результатов предсказания классификации от модели
  -----
  :param y_pred: предугаданные шансы для каждого класса
  :return: нормализованные вероятности, в сумме даёт 1
  """
    result = np.zeros_like(y_pred)
    for i, l in enumerate(y_pred):
        normalized = (l - np.min(l)) / (np.max(l) - np.min(l))
        result[i] = normalized / np.sum(normalized)
    return result


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])


class MLP:
    """ Классический многослойный персептрон"""

    def __init__(self, architecture, lr=0.01, is_need_f_final=False, fn=(sigmoid, dsigmoid)):
        self.depth = len(architecture) - 1
        self.lr = lr
        # Требуется ли активационная функция на выходном слое
        self._is_need_func_final = is_need_f_final

        self.activation_fn = fn[0]
        self.activation_dfn = fn[1]

        self.error_fn = mse
        self.error_dfn = dmse

        # TODO: добавить сохранение весов и смещения, чтобы сохранять успешное обучение между запусками
        # Веса (k, i, j - k: номер слоя 0 = первый скрытый слой; i - конкретный нейрон; j - нейрон предыдущего слоя)
        self.W = self._init_weights(architecture)
        # Смещение (или порог активации)
        # (k, i, j - k: номер слоя 0 = первый скрытый слой; i - конкретный нейрон; j=1)
        self.b = self._init_biases(architecture)

        # Прямое направление
        # Сумматор
        self.z = [None] * (self.depth + 1)
        # Результат функции активации (кроме последнего слоя) (k: номер слоя: 0 - вход, 1 - первый скрытый слой; j количество нейронов на слое; l - количество данных на батч
        self.a = [None] * (self.depth + 1)

        # Дельта правило (при инициализации заполняются 0)
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

        # glorot uniform init

    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        return [rng.uniform(-limit, limit + 1e-5, size=(arch[i + 1], arch[i])) for i in range(self.depth)]

    def _init_biases(self, arch):
        return [rng.random((arch[i + 1], 1)) * 2 - 1 for i in range(self.depth)]

    def save_weight(self, filename):
        np.save(filename + '_W', self.W)
        np.save(filename + '_b', self.b)

    def set_weights(self, new_W):
        self.W = new_W

    def load_weights(self, filename):
        new_W = np.load(filename + '_W.npy', allow_pickle=True)
        self.set_weights(new_W)

    def set_biases(self, new_b):
        self.b = new_b

    def load_biases(self, filename):
        new_b = np.load(filename + '_b.npy', allow_pickle=True)
        self.set_biases(new_b)

    def compute_loss(self, X, y):
        # X[l][in]
        y_pred = self.predict(X).reshape(y.shape)
        return self.error_fn(y_pred, y)

    # Расчет прямого прохождения для нескольких X
    def _feedforward(self, X):
        # W[0].shape[1] - количество входов = net_in
        # -1; X.shape[0] - количество значений в батче
        self.a[0] = X.T.reshape(self.W[0].shape[1], -1)
        # рассчитываем скрытые и выходной слои
        for k in range(self.depth):
            # перемножаются W [i (текущий слой), j (предыдущий)] * a[j (количество нейронов), l (количество в батче)]
            self.z[k + 1] = np.matmul(self.W[k], self.a[k]) + self.b[k]
            # Выполняется только для скрытых слоев либо если передан _is_need_func_final
            # TODO : добавить отдельную функцию активации для выходного слоя
            if k != self.depth - 1 or self._is_need_func_final:
                self.a[k + 1] = self.activation_fn(self.z[k + 1])
            else:
                self.a[-1] = self.z[-1]

    # Расчет локального градиента при обратном распространении ошибок
    def _backprop(self, y, batch_size=32):
        # TODO сделать delta для выходящего слоя при _is_need_func_final и конкретной функции
        # a[j, l], y[l, in] -> y[in, l]
        delta = self.error_dfn(self.a[-1], y.T)
        if self._is_need_func_final:
            delta *= self.activation_dfn(self.z[-1])
        for k in range(self.depth - 1, -1, -1):
            # Идем в обратном порядке. k - текущий слой, k+1 уже подсчитанный. j - количество нейронов в текущем слое. i - количество нейронов в k+1 слое
            if k != self.depth - 1:
                # todo проверить как работает на выходном слое
                # f(z[i][l]) * (dot (W[j][i], delta[i][l]) получаем [j][l], так как для W[k].shape(i) = z[k].shape(j)
                delta = self.activation_dfn(self.z[k + 1]) * np.matmul(self.W[k + 1].T, delta)
            # dot(delta[j][l], a[l][j]) должны получить dW[i, j]
            self.dW[k] = np.matmul(delta, self.a[k].T)
            # delta[j][1] sum l
            self.db[k] = np.sum(delta, axis=1, keepdims=True)

    # simple sgd
    def _update_params(self, lr=1e-2):
        for k in range(self.depth):
            self.W[k] -= lr * self.dW[k]
            self.b[k] -= lr * self.db[k]

    def train(self, X, y, lr=None, X_test=None, y_test=None, epochs=50, batch_size=32):
        """
    Обучение модели.

    -------
    :param X: Размерности l * in, где l количество входящих данных, in - количество входов
    :param y: Размерности l * out.
    :param lr:  learning rate, по умолчанию берет значение из MLP.lr
    :param X_test, y_test: для анализа переобучения. По дефолту None
    :param y_test:
    :param epochs: Количество эпох, по умолчанию 50.
    :param batch_size: Количество данных в одном батче. По умолчанию 32.
    :return: Возвращает вектора loss и dW(mean каждого слоя) по эпохам
    """
        lr = lr if lr is not None else self.lr

        epoch_losses = np.array([])
        epoch_grad_w = np.zeros((len(self.dW), epochs))
        epoch_loss_test = None
        if X_test is not None and y_test is not None:
            epoch_loss_test = np.array([])
        dataset = list(zip(X, y))
        for i in range(epochs):
            rng.shuffle(dataset)
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                self._feedforward(X_batch)
                self._backprop(y_batch)
                self._update_params(lr=lr)

            epoch_losses = np.append(epoch_losses, self.compute_loss(X, y))
            if epoch_loss_test is not None:
                epoch_loss_test = np.append(epoch_loss_test, self.compute_loss(X_test, y_test))
            for k in range(len(self.dW)):
                epoch_grad_w[k][i] = np.mean(self.dW[k])
        return epoch_losses, epoch_grad_w, epoch_loss_test

    def predict_proba(self, X, is_classification=False):
        """
    Предсказать значение на уже обученной модели.

    ------
    :param X:На вход X размерности l * in, где l - количество полученных данных, in - количество входящих нейронов.
    :param is_classification: является ли это предсказание класса? Если да, то возвращает вероятность к конкретному классу
    :return: Возвращает a размерности l * out
    """
        a = X.T.reshape(self.W[0].shape[1], -1)
        # compute hidden and output layers
        for i in range(self.depth):
            a = np.matmul(self.W[i], a) + self.b[i]
            if i != self.depth - 1 or self._is_need_func_final:
                a = self.activation_fn(a)
        a = a.T
        if is_classification:
            a = normalize(a)
        return a

    def predict(self, X, is_classification=False):
        """
    Предсказать значение на уже обученной модели.

    ------
    :param X:На вход X размерности l * out, где l - количество полученных данных, out - количество выходящих нейронов.
    :param is_classification: является ли это предсказание класса? Если да, то возвращает вероятность к конкретному классу
    :return: Возвращает a размерности l * out (если is_classification=True то out=1)
    """
        res = self.predict_proba(X, is_classification)
        if is_classification:
            return np.array([np.argmax(l) for l in res])
        else:
            return res


def roc_curve_manual(y_true, y_pred_proba, class_label):
    """

  :param y_true: изначальные y размерности (l,)
  :param y_pred_proba: результат расчета вероятностей для классов размерности (l, k)
  :param class_label: класс, для которого рассчитываем roc
  :return:
  """
    y_true = y_true.reshape(-1)
    y_pred_proba_class = y_pred_proba[:, class_label].copy()  # Берем вероятности только для class_label
    thresholds = np.sort(np.unique(y_pred_proba_class))[::-1]  # пороги сортируются по убыванию
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_pred_proba_class >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == class_label))
        fn = np.sum((y_pred == 0) & (y_true == class_label))
        fp = np.sum((y_pred == 1) & (y_true != class_label))
        tn = np.sum((y_pred == 0) & (y_true != class_label))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list


class Metrics:

    def __init__(self, y_pred, y):
        self.confusion_matrix = self._init_confusion_matrix(y_pred, y)
        self.accuracy = np.trace(self.confusion_matrix) \
                        / np.sum(self.confusion_matrix)
        confusion_matrix = self.confusion_matrix
        # Инициализируем списки для precision, recall и F1 для каждого класса
        precision = []
        recall = []
        f1_score = []

        # Рассчитываем метрики для каждого класса
        for i in range(len(self.confusion_matrix)):
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i, :]) - TP
            TN = np.sum(confusion_matrix) - (TP + FP + FN)

            # Precision, Recall, F1 для класса i
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            precision.append(prec)
            recall.append(rec)
            f1_score.append(f1)

            print(f"Class {i}: Precision = {prec:.4f}, Recall = {rec:.4f}, F1-Score = {f1:.4f}")
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score

    def _init_confusion_matrix(self, y_pred, y):
        n = max(np.max(y_pred), np.max(y)) + 1
        matrix = np.zeros((n, n))
        for i in range(len(y_pred)):
            pred_class = y_pred[i]
            class_label = y[i]
            matrix[pred_class, class_label] += 1
        return matrix


def test_forward():
    test = MLP((5, 3))
    print(test.W)
    print(test.b)
    inp = np.array([1, 0, 1, 0, 1]).reshape((1, 5))
    result = test.predict_proba(inp)
    print(result)

    """
  [array([[ 0.76634006, -0.33404309, -0.42451161, -0.48043658,  0.49572811],
       [-0.49067389,  0.5295419 ,  0.40298306, -0.13548592, -0.01413979],
       [ 0.70883497,  0.50398643, -0.59962183,  0.43127379, -0.16798996]])]
[array([[ 0.98281432],
       [-0.57678455],
       [-0.17904883]])]
[[ 1.82037088 -0.67861517 -0.23782566]]
"""


def test_backprop():
    test = MLP((5, 3))
    print(test.W)
    print(test.b)
    in_s = np.array([1, 0, 1, 0, 1]).reshape((1, 5))
    result = test.predict_proba(in_s)
    out_s = np.array([0, 0, 1]).reshape(1, 3)
    loss = test.train(in_s, out_s, epochs=1, lr=0.01)
    print(loss)
    result = test.predict_proba(in_s)
    print(f"result2: {result}")
    print(f"weights: {test.W}")
    print(f"dW: {test.dW}")

    """[array([[ 0.76634006, -0.33404309, -0.42451161, -0.48043658,  0.49572811],
       [-0.49067389,  0.5295419 ,  0.40298306, -0.13548592, -0.01413979],
       [ 0.70883497,  0.50398643, -0.59962183,  0.43127379, -0.16798996]])]
[array([[ 0.98281432],
       [-0.57678455],
       [-0.17904883]])]
(array([1.5600045]), array([[0.36157202]]), None)
result2: [[ 1.67474121 -0.62432596 -0.21879961]]
weights: [array([[ 7.29932638e-01, -3.34043087e-01, -4.60919030e-01,
        -4.80436582e-01,  4.59320696e-01],
       [-4.77101589e-01,  5.29541895e-01,  4.16555361e-01,
        -1.35485919e-01, -5.67486201e-04],
       [ 7.13591481e-01,  5.03986427e-01, -5.94865315e-01,
         4.31273786e-01, -1.63233450e-01]])]
dW: [array([[ 3.64074176,  0.        ,  3.64074176,  0.        ,  3.64074176],
       [-1.35723035,  0.        , -1.35723035,  0.        , -1.35723035],
       [-0.47565132,  0.        , -0.47565132,  0.        , -0.47565132]])]"""


def test_bacprop_2layers():
    test = MLP((5, 4, 3))
    new_W = [np.array([[0.72251287, -0.31493867, -0.40023324, -0.45295983, 0.4673774],
                       [-0.46261166, 0.49925735, 0.37993649, -0.12773711, -0.01333084],
                       [0.66829654, 0.47516341, -0.56532879, 0.40660925, -0.15838223],
                       [0.80247445, -0.47094049, -0.14618866, 0.72389501, 0.13097715]]),
             np.array([[-0.74982772, -0.25898291, -0.75532695, -0.61120268],
                       [-0.02611936, -0.2681479, 0.11507216, -0.29015443],
                       [0.103804, 0.2647263, 0.65609751, -0.86375996]])]
    new_b = [np.array([[0.47150135],
                       [0.20950201],
                       [-0.53731956],
                       [-0.89091086]]),
             np.array([[0.50152955],
                       [-0.25262151],
                       [-0.50155969]])]
    print(f"old_shape ={test.W[0].shape}, new_shape={new_W[0].shape}")
    test.set_weights(new_W)
    test.set_biases(new_b)
    print(f"init_weights: {test.W}")
    print(f"init_bias: {test.b}")
    in_s = np.array([1, 0, 1, 0, 1]).reshape((1, 5))
    result = test.predict_proba(in_s)
    print(f"result: {result}")
    out_s = np.array([0, 0, 1]).reshape(1, 3)
    loss = test.train(in_s, out_s, epochs=1, lr=0.01)
    # print(f"loss {loss}")
    result = test.predict_proba(in_s)
    print(f"result2: {result}")
    print(f"weights: {test.W}")
    print(f"dW: {test.dW}")

    """
    old_shape =(4, 5), new_shape=(4, 5)
init_weights: [array([[ 0.72251287, -0.31493867, -0.40023324, -0.45295983,  0.4673774 ],
       [-0.46261166,  0.49925735,  0.37993649, -0.12773711, -0.01333084],
       [ 0.66829654,  0.47516341, -0.56532879,  0.40660925, -0.15838223],
       [ 0.80247445, -0.47094049, -0.14618866,  0.72389501,  0.13097715]]), array([[-0.74982772, -0.25898291, -0.75532695, -0.61120268],
       [-0.02611936, -0.2681479 ,  0.11507216, -0.29015443],
       [ 0.103804  ,  0.2647263 ,  0.65609751, -0.86375996]])]
init_bias: [array([[ 0.47150135],
       [ 0.20950201],
       [-0.53731956],
       [-0.89091086]]), array([[ 0.50152955],
       [-0.25262151],
       [-0.50155969]])]
result: [[-0.77826756 -0.51124748 -0.45674897]]
result2: [[-0.7384139  -0.48552645 -0.38247212]]
weights: [array([[ 0.72097935, -0.31493867, -0.40176676, -0.45295983,  0.46584388],
       [-0.46237746,  0.49925735,  0.38017069, -0.12773711, -0.01309664],
       [ 0.67025334,  0.47516341, -0.56337199,  0.40660925, -0.15642543],
       [ 0.79308817, -0.47094049, -0.15557494,  0.72389501,  0.12159087]]), 
     array([[-0.7376988 , -0.25075906, -0.74978556, -0.60382297],
       [-0.01815182, -0.26274561,  0.11871232, -0.28530667],
       [ 0.12650672,  0.28011958,  0.66646979, -0.84994674]])]
dW: [array([[ 0.15335161,  0.        ,  0.15335161,  0.        ,  0.15335161],
       [-0.02342041,  0.        , -0.02342041,  0.        , -0.02342041],
       [-0.19568023,  0.        , -0.19568023,  0.        , -0.19568023],
       [ 0.93862754,  0.        ,  0.93862754,  0.        ,  0.93862754]]), 
     array([[-1.21289178, -0.82238534, -0.55413862, -0.73797072],
       [-0.7967541 , -0.54022865, -0.36401617, -0.4847763 ],
       [-2.27027175, -1.53932794, -1.03722795, -1.38132199]])]"""

    """dmse=[[-1.55653512], [-1.02249496], [-2.91349795]]"""

    """delta from lay2:[[-1.55653512], [-1.02249496], [-2.91349795]]"""
    """delta from lay1:[[ 0.15335161], [-0.02342041], [-0.19568023], [ 0.93862754]]"""
