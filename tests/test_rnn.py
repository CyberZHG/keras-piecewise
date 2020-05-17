import unittest
import numpy as np
from keras_piecewise.backend import keras
from keras_piecewise.backend import TF_KERAS
from keras_piecewise import Piecewise


class TestRNN(unittest.TestCase):

    @staticmethod
    def _build_model(input_shape, layer, piece_num):
        data_input = keras.layers.Input(shape=input_shape)
        position_input = keras.layers.Input(shape=(piece_num,))
        pool_layer = Piecewise(
            layer=layer,
        )([data_input, position_input])
        model = keras.models.Model(inputs=[data_input, position_input], outputs=pool_layer)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def test_lstm(self):
        if not TF_KERAS:
            return
        data = [[[1, 3, 2, 5], [7, 4, 2, 3], [0, 1, 2, 2], [4, 7, 2, 5]]]
        positions = [[1, 2, 4]]
        model = self._build_model(
            input_shape=(None, 4),
            layer=keras.layers.LSTM(units=3),
            piece_num=len(positions[0]),
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)])
        self.assertEqual((1, 3, 3), predicts.shape)
