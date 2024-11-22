import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def load_data(sequence_len: int = 10):
    """

    :param sequence_len: длина последовательности t
    :return:
    """

    df = pd.read_csv('Steel_industry_data.csv')
    # df = df.sort_values(by="date")
    df = df.drop(columns=["date"])
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        df[column_name] = le.fit_transform(df[column_name])

    df = _preprocessing(df)

    Y = df[:, 0]
    X = df[:, 1:]

    # X, Y = _preprocessing(df.iloc[:, 1:-1], df.iloc[:, -1:].values.ravel())
    X, y = _split_for_rnn(X, Y, sequence_len)
    return X, y.reshape(-1, sequence_len, 1)


def _preprocessing(data):
    # 3. Масштабирование данных
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def _split_for_rnn(features, labels, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])  # Окно признаков
        y.append(labels[i:i + seq_len])  # Метка последнего шага
    return np.array(X), np.array(y)


def test_init():
    load_data(10)