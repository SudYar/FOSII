import enum

from Layer import Layer
from Dense import Dense
from Convolution2D import Convolution2D
from Maxpooling import Maxpooling
from Flatten import Flatten


class LayerTypes(enum.Enum):

    DENSE = "Dense"
    CONVOLUTIONAL = "Convolutional"
    POOLING = "MaxPooling"
    FLATTEN = "Flatten"


def create_layer(layer_type: LayerTypes, **kwargs) -> Layer:
    layer_classes = {
        LayerTypes.CONVOLUTIONAL: Convolution2D,
        LayerTypes.FLATTEN: Flatten,
        LayerTypes.POOLING: Maxpooling,
        LayerTypes.DENSE: Dense
    }
    layer_class = layer_classes.get(layer_type)

    if layer_class is not None:
        return layer_class(**kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
