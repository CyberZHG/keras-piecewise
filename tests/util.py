from keras_piecewise.backend import keras
from keras_piecewise.backend import backend as K


class MaxPool1D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaxPool1D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.max(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]


class AvePool1D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AvePool1D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.cast(K.sum(inputs, axis=1), K.floatx()) / K.cast(K.shape(inputs)[1], K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[2:]


class MaxPool2D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)

    def call(self, inputs):
        return K.max(K.max(inputs, axis=1), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + input_shape[3:]
