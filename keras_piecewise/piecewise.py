import keras
import keras.backend as K


class Piecewise(keras.layers.Layer):

    POOL_TYPE_MAX = 'max'
    POOL_TYPE_AVERAGE = 'average'

    def __init__(self,
                 piece_num,
                 pool_type=POOL_TYPE_MAX,
                 **kwargs):
        self.piece_num = piece_num
        self.pool_type = pool_type
        self.supports_masking = True
        super(Piecewise, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'piece_num': self.piece_num,
            'pool_type': self.pool_type,
        }
        base_config = super(Piecewise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(Piecewise, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        inputs, positions = inputs
        return K.map_fn(
            lambda i: self._call_sample(inputs, positions, i),
            K.arange(K.shape(inputs)[0]),
            dtype=K.floatx(),
        )

    def _call_sample(self, inputs, positions, index):
        inputs = inputs[index]
        positions = positions[index]
        return K.map_fn(
            lambda i: self._call_piece(inputs, positions, i),
            K.arange(self.piece_num),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, positions, index):
        piece = inputs[K.switch(K.equal(index, 0), 0, positions[index - 1]):positions[index]]
        return K.switch(
            K.equal(K.shape(piece)[0], 0),
            lambda: self._pool_empty(inputs),
            lambda: self._pool_type(piece),
        )

    def _pool_empty(self, piece):
        return K.zeros_like(K.max(piece, axis=0))

    def _pool_type(self, piece):
        if callable(self.pool_type):
            return self.pool_type(piece)
        if self.pool_type == self.POOL_TYPE_MAX:
            return K.max(piece, axis=0)
        if self.pool_type == self.POOL_TYPE_AVERAGE:
            return K.sum(piece, axis=0) / K.cast(K.shape(piece)[0], K.floatx())
        raise NotImplementedError('No implementation for pooling type : ' + self.pool_type)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.piece_num) + tuple(input_shape[0][2:])

    def compute_mask(self, inputs, mask=None):
        return None
