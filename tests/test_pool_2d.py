import unittest
import os
import tempfile
import numpy as np
from keras_piecewise.backend import keras
from keras_piecewise import Piecewise2D
from .util import MaxPool2D


class TestPool2D(unittest.TestCase):

    @staticmethod
    def _build_model(input_shape, layer, row_num, col_num, pos_type=Piecewise2D.POS_TYPE_SEGMENTS):
        data_input = keras.layers.Input(shape=input_shape)
        row_input = keras.layers.Input(shape=(row_num,))
        col_input = keras.layers.Input(shape=(col_num,))
        pool_layer = Piecewise2D(
            layer=layer,
            pos_type=pos_type,
        )([data_input, row_input, col_input])
        model = keras.models.Model(inputs=[data_input, row_input, col_input], outputs=pool_layer)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
        model.summary()
        return model

    def test_max_2d(self):
        data = [
            [
                [1, 3, 5, 2],
                [2, 5, 6, 1],
                [7, 1, 5, 3],
                [7, 2, 2, 4],
            ],
            [
                [1, 3, 5, 2],
                [2, 5, 6, 1],
                [7, 1, 5, 3],
                [7, 2, 2, 4],
            ],
        ]
        rows = [
            [2, 4],
            [3, 4],
        ]
        cols = [
            [1, 2, 4],
            [1, 3, 4],
        ]
        model = self._build_model(
            input_shape=(None, None),
            layer=MaxPool2D(),
            row_num=len(rows[0]),
            col_num=len(cols[0]),
        )
        predicts = model.predict([np.asarray(data), np.asarray(rows), np.asarray(cols)]).tolist()
        expected = [
            [
                [2.0, 5.0, 6.0],
                [7.0, 2.0, 5.0],
            ],
            [
                [7.0, 6.0, 3.0],
                [7.0, 2.0, 4.0],
            ],
        ]
        self.assertEqual(expected, predicts)
        cols = [
            [1, 2, 0, 4],
            [1, 3, 2, 4],
        ]
        model = self._build_model(
            input_shape=(None, None),
            layer=MaxPool2D(),
            row_num=len(rows[0]),
            col_num=len(cols[0]),
            pos_type=Piecewise2D.POS_TYPE_PAIRS,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_piece_test_save_load_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Piecewise2D': Piecewise2D,
            'MaxPool2D': MaxPool2D,
        })
        predicts = model.predict([np.asarray(data), np.asarray(rows), np.asarray(cols)]).tolist()
        expected = [
            [[2.0, 7.0]],
            [[2.0, 4.0]],
        ]
        self.assertEqual(expected, predicts)

    def test_pos_type_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self._build_model(
                input_shape=(None,),
                layer=MaxPool2D(),
                row_num=13,
                col_num=17,
                pos_type='whatever',
            )
