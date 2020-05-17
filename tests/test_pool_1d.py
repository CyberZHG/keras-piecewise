import os
import tempfile
import unittest
import numpy as np
from keras_piecewise.backend import keras
from keras_piecewise import Piecewise
from .util import MaxPool1D, AvePool1D


class TestPool1D(unittest.TestCase):

    @staticmethod
    def _build_model(input_shape, layer, piece_num, pos_type=Piecewise.POS_TYPE_SEGMENTS):
        data_input = keras.layers.Input(shape=input_shape)
        position_input = keras.layers.Input(shape=(piece_num,))
        pool_layer = Piecewise(
            layer=layer,
            pos_type=pos_type,
        )([data_input, position_input])
        model = keras.models.Model(inputs=[data_input, position_input], outputs=pool_layer)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
        model.summary()
        return model

    def test_max_2d(self):
        data = [
            [1, 3, 5, 2],
            [1, 3, 5, 2],
        ]
        positions = [
            [2, 4],
            [3, 4],
        ]
        model = self._build_model(
            input_shape=(None,),
            layer=MaxPool1D(),
            piece_num=len(positions[0]),
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [3.0, 5.0],
            [5.0, 2.0],
        ]
        self.assertEqual(expected, predicts)

    def test_average_3d(self):
        data = [[[1, 3, 2, 5], [7, 9, 2, 3], [0, 1, 7, 2], [4, 7, 2, 5]]]
        positions = [[1, 3, 4]]
        model = self._build_model(
            input_shape=(None, None),
            layer=AvePool1D(),
            piece_num=len(positions[0]),
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [[
            [1.0, 3.0, 2.0, 5.0],
            [3.5, 5.0, 4.5, 2.5],
            [4.0, 7.0, 2.0, 5.0],
        ]]
        self.assertEqual(expected, predicts)

    def test_empty_interval(self):
        data = [[[1, 3, 2, 5], [7, 4, 2, 3], [0, 1, 2, 2], [4, 7, 2, 5]]]
        positions = [[2, 2, 4]]
        model = self._build_model(
            input_shape=(None, None),
            layer=MaxPool1D(),
            piece_num=len(positions[0]),
        )
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [[
            [7.0, 4.0, 2.0, 5.0],
            [float('-inf'), float('-inf'), float('-inf'), float('-inf')],
            [4.0, 7.0, 2.0, 5.0],
        ]]
        self.assertEqual(expected, predicts)

    def test_save_load(self):
        data = [[[1, 5, 2, 4], [7, 9, 2, 3], [0, 1, 7, 2], [4, 8, 2, 5]]]
        positions = [[2, 4]]
        model = self._build_model(
            input_shape=(None, None),
            layer=AvePool1D(),
            piece_num=len(positions[0]),
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_piece_test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Piecewise': Piecewise,
            'AvePool1D': AvePool1D,
        })
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [[
            [4.0, 7.0, 2.0, 3.5],
            [2.0, 4.5, 4.5, 3.5],
        ]]
        self.assertEqual(expected, predicts)

    def test_max_2d_pair(self):
        data = [
            [1, 3, 9, 2],
            [1, 9, 8, 2],
        ]
        positions = [
            [1, 4, 0, 2],
            [2, 4, 2, 3],
        ]
        model = self._build_model(
            input_shape=(None,),
            layer=MaxPool1D(),
            piece_num=len(positions[0]),
            pos_type=Piecewise.POS_TYPE_PAIRS,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_piece_test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Piecewise': Piecewise,
            'MaxPool1D': MaxPool1D,
        })
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [9.0, 3.0],
            [8.0, 8.0],
        ]
        self.assertEqual(expected, predicts)

    def test_pos_type_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._build_model(
                input_shape=(None,),
                layer=MaxPool1D(),
                piece_num=12,
                pos_type='whatever',
            )
