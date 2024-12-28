import enum

from Layer import Layer
from RNNSimple import RNNSimple
from LSTMLayer import LSTMLayer
from GRULayer import GRULayer
from RNN import RNN
from src.main.python.sudyar.jupyter.Lab3.GRU import GRU


class RecurrentLayerTypes(enum.Enum):
    RNNSIMPLE = "RNNSimple"
    LSTM = "LSTM"
    GRU = "GRU"
    RNN = "RNN with graph"
    GRUgr = "GRU with graph"


def create_layer(layer_type: RecurrentLayerTypes, **kwargs) -> Layer:
    layer_classes = {
        RecurrentLayerTypes.RNNSIMPLE: RNNSimple,
        RecurrentLayerTypes.LSTM: LSTMLayer,
        RecurrentLayerTypes.GRU: GRULayer,
        RecurrentLayerTypes.RNN: RNN,
        RecurrentLayerTypes.GRUgr: GRU
    }
    layer_class = layer_classes.get(layer_type)

    if layer_class is not None:
        return layer_class(**kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
