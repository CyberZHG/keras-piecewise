
Keras Piecewise
===============


.. image:: https://travis-ci.org/CyberZHG/keras-piecewise.svg
   :target: https://travis-ci.org/CyberZHG/keras-piecewise
   :alt: Travis


.. image:: https://coveralls.io/repos/github/CyberZHG/keras-piecewise/badge.svg?branch=master
   :target: https://coveralls.io/github/CyberZHG/keras-piecewise
   :alt: Coverage


.. image:: https://img.shields.io/pypi/pyversions/keras-piecewise.svg
   :target: https://pypi.org/project/keras-piecewise/
   :alt: PyPI


A wrapper layer for splitting and accumulating sequential data.


.. image:: https://user-images.githubusercontent.com/853842/45488448-07e08e80-b794-11e8-8b67-ae650aa017b5.png
   :target: https://user-images.githubusercontent.com/853842/45488448-07e08e80-b794-11e8-8b67-ae650aa017b5.png
   :alt: 


Install
-------

.. code-block:: bash

   pip install keras-piecewise-pooling

Usage
-----

.. code-block:: python

   import keras
   import numpy as np
   from keras_piecewise import Piecewise


   class AvePool1D(keras.layers.Layer):

       def __init__(self, **kwargs):
           super(AvePool1D, self).__init__(**kwargs)

       def call(self, inputs):
           return K.sum(inputs, axis=1) / K.cast(K.shape(inputs)[1], K.floatx())

       def compute_output_shape(self, input_shape):
           return (input_shape[0],) + input_shape[1:]


   data = [[[1, 3, 2, 5], [7, 9, 2, 3], [0, 1, 7, 2], [4, 7, 2, 5]]]
   positions = [[1, 3, 4]]
   piece_num = len(positions[0])

   data_input = keras.layers.Input(shape=(None, None))
   position_input = keras.layers.Input(shape=(piece_num,), dtype='int32')
   pool_layer = Piecewise(
       layer=AvePool1D(),
       piece_num=piece_num,
   )([data_input, position_input])
   model = keras.models.Model(inputs=[data_input, position_input], outputs=pool_layer)
   model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
   model.summary()

   print(model.predict([np.asarray(data), np.asarray(positions)]).tolist())
   # The result will be:
   # [[
         [1.0, 3.0, 2.0, 5.0],
         [3.5, 5.0, 4.5, 2.5],
         [4.0, 7.0, 2.0, 5.0],
   # ]]
