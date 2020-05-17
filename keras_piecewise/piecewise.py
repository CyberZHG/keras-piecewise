import inspect

from .backend import keras
from .backend import backend as K

__all__ = ['Piecewise', 'Piecewise2D']


try:
    Wrapper = keras.layers.Wrapper
except AttributeError:
    Wrapper = keras.layers.wrappers.Wrapper


def _has_arg(name, func):
    signature = inspect.signature(func)
    parameter = signature.parameters.get(name)
    if parameter is None:
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False
    return True


class Piecewise(Wrapper):

    POS_TYPE_SEGMENTS = 'segments'
    POS_TYPE_PAIRS = 'pairs'

    def __init__(self,
                 layer,
                 pos_type=POS_TYPE_SEGMENTS,
                 **kwargs):
        """Initialize the wrapper layer.

        :param layer: The layer that process the inner pieces.
        :param piece_num: The fixed number of pieces for each data.
        :param pos_type: The type of position input.
        :param kwargs: Arguments for parent.
        """
        super(Piecewise, self).__init__(layer, **kwargs)
        if pos_type not in {self.POS_TYPE_SEGMENTS, self.POS_TYPE_PAIRS}:
            raise NotImplementedError('No implementation for position type : %s' % pos_type)
        self.layer = layer
        self.pos_type = pos_type
        self.supports_masking = True
        self.zeros = None

    def get_config(self):
        config = {'pos_type': self.pos_type}
        base_config = super(Piecewise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape[0])
            self.layer.built = True
        self.zeros = K.zeros((1, 1), dtype='int32', name='zeros')
        super(Piecewise, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        inputs, positions = inputs
        if K.dtype(positions) != 'int32':
            positions = K.cast(positions, 'int32')
        if self.pos_type == self.POS_TYPE_SEGMENTS:
            positions = K.concatenate([K.tile(self.zeros, (K.shape(inputs)[0], 1)), positions], axis=1)
        inputs = K.cast(inputs, K.floatx())
        return K.map_fn(
            lambda i: self._call_sample(inputs, training, positions, i),
            K.arange(0, K.shape(inputs)[0]),
            dtype=K.floatx(),
        )

    def _call_sample(self, inputs, training, positions, index):
        step = 1
        if self.pos_type == self.POS_TYPE_PAIRS:
            step = 2
        inputs = inputs[index]
        positions = positions[index]
        return K.map_fn(
            lambda i: self._call_piece(inputs, training, positions, i),
            K.arange(1, K.shape(positions)[0], step),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, training, positions, index):
        piece = inputs[positions[index - 1]:positions[index]]
        kwargs = {'inputs': K.expand_dims(piece, axis=0)}
        if _has_arg('training', self.layer.call):
            kwargs['training'] = training
        return K.squeeze(self.layer.call(**kwargs), axis=0)

    def compute_output_shape(self, input_shape):
        pos_len = input_shape[1][1]
        if self.pos_type == self.POS_TYPE_PAIRS:
            pos_len //= 2
        child_output_shape = self.layer.compute_output_shape(input_shape[0])
        return (input_shape[0][0], pos_len) + child_output_shape[1:]

    def compute_mask(self, inputs, mask=None):
        return None


class Piecewise2D(Wrapper):

    POS_TYPE_SEGMENTS = 'segments'
    POS_TYPE_PAIRS = 'pairs'

    def __init__(self,
                 layer,
                 pos_type=POS_TYPE_SEGMENTS,
                 **kwargs):
        """Initialize the wrapper layer.

        :param layer: The layer that process the inner pieces.
        :param piece_num: The fixed number of pieces for each data.
        :param pos_type: The type of position input.
        :param kwargs: Arguments for parent.
        """
        super(Piecewise2D, self).__init__(layer, **kwargs)
        self.layer = layer
        self.supports_masking = True
        if pos_type not in {self.POS_TYPE_SEGMENTS, self.POS_TYPE_PAIRS}:
            raise NotImplementedError('No implementation for position type : %s' % pos_type)
        self.pos_type = pos_type
        self.zeros = None

    def get_config(self):
        config = {'pos_type': self.pos_type}
        base_config = super(Piecewise2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape[0])
            self.layer.built = True
        self.zeros = K.zeros((1, 1), dtype='int32', name='zeros')
        super(Piecewise2D, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        inputs, rows, cols = inputs
        if K.dtype(rows) != 'int32':
            rows = K.cast(rows, 'int32')
        if K.dtype(cols) != 'int32':
            cols = K.cast(cols, 'int32')
        if self.pos_type == self.POS_TYPE_SEGMENTS:
            rows = K.concatenate([K.tile(self.zeros, (K.shape(inputs)[0], 1)), rows], axis=1)
            cols = K.concatenate([K.tile(self.zeros, (K.shape(inputs)[0], 1)), cols], axis=1)
        inputs = K.cast(inputs, K.floatx())
        return K.map_fn(
            lambda i: self._call_sample(inputs, training, rows, cols, i),
            K.arange(0, K.shape(inputs)[0]),
            dtype=K.floatx(),
        )

    def _call_sample(self, inputs, training, rows, cols, index):
        step = 1
        if self.pos_type == self.POS_TYPE_PAIRS:
            step = 2
        inputs, rows, cols = inputs[index], rows[index], cols[index]
        return K.map_fn(
            lambda row: self._call_rows(inputs, training, rows, cols, row),
            K.arange(1, K.shape(rows)[0], step),
            dtype=K.floatx(),
        )

    def _call_rows(self, inputs, training, rows, cols, row):
        step = 1
        if self.pos_type == self.POS_TYPE_PAIRS:
            step = 2
        return K.map_fn(
            lambda col: self._call_piece(inputs, training, rows, cols, row, col),
            K.arange(1, K.shape(cols)[0], step),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, training, rows, cols, row, col):
        piece = inputs[rows[row - 1]:rows[row], cols[col - 1]:cols[col]]
        kwargs = {'inputs': K.expand_dims(piece, axis=0)}
        if _has_arg('training', self.layer.call):
            kwargs['training'] = training
        return K.squeeze(self.layer.call(**kwargs), axis=0)

    def compute_output_shape(self, input_shape):
        pos_row_len, pos_col_len = input_shape[1][1], input_shape[2][1]
        if self.pos_type == self.POS_TYPE_PAIRS:
            pos_row_len //= 2
            pos_col_len //= 2
        child_output_shape = self.layer.compute_output_shape(input_shape[0])
        return (input_shape[0][0], pos_row_len, pos_col_len) + child_output_shape[1:]

    def compute_mask(self, inputs, mask=None):
        return None
