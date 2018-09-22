import keras
import keras.backend as K


class Piecewise(keras.layers.Wrapper):

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
        self.layer = layer
        if pos_type not in {self.POS_TYPE_SEGMENTS, self.POS_TYPE_PAIRS}:
            raise NotImplementedError('No implementation for position type : %s' % pos_type)
        self.pos_type = pos_type
        self.supports_masking = True
        super(Piecewise, self).__init__(layer, **kwargs)

    def get_config(self):
        config = {'pos_type': self.pos_type}
        base_config = super(Piecewise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = list(map(lambda x: keras.engine.InputSpec(shape=x), input_shape))
        if not self.layer.built:
            self.layer.build(input_shape[0])
            self.layer.built = True
        super(Piecewise, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        inputs, positions = inputs
        if K.dtype(positions) != 'int32':
            positions = K.cast(positions, 'int32')
        if self.pos_type == self.POS_TYPE_SEGMENTS:
            positions = K.concatenate([K.zeros((K.shape(inputs)[0], 1), dtype='int32'), positions], axis=1)
        return K.map_fn(
            lambda i: self._call_sample(inputs, positions, i),
            K.arange(K.shape(inputs)[0]),
            dtype=K.floatx(),
        )

    def _call_sample(self, inputs, positions, index):
        step = 1
        if self.pos_type == self.POS_TYPE_PAIRS:
            step = 2
        inputs = inputs[index]
        positions = positions[index]
        return K.map_fn(
            lambda i: self._call_piece(inputs, positions, i),
            K.arange(1, K.shape(positions)[0], step),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, positions, index):
        piece = inputs[positions[index - 1]:positions[index]]
        return K.squeeze(self.layer.call(inputs=K.expand_dims(piece, axis=0)), axis=0)

    def compute_output_shape(self, input_shape):
        pos_len = input_shape[1][1]
        if self.pos_type == self.POS_TYPE_PAIRS:
            pos_len //= 2
        child_output_shape = self.layer.compute_output_shape(input_shape[0])
        return (input_shape[0][0], pos_len) + child_output_shape[1:]

    def compute_mask(self, inputs, mask=None):
        return None


class Piecewise2D(keras.layers.Wrapper):

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
        self.layer = layer
        self.supports_masking = True
        if pos_type not in {self.POS_TYPE_SEGMENTS, self.POS_TYPE_PAIRS}:
            raise NotImplementedError('No implementation for position type : %s' % pos_type)
        self.pos_type = pos_type
        super(Piecewise2D, self).__init__(layer, **kwargs)

    def get_config(self):
        config = {'pos_type': self.pos_type}
        base_config = super(Piecewise2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.input_spec = list(map(lambda x: keras.engine.InputSpec(shape=x), input_shape))
        if not self.layer.built:
            self.layer.build(input_shape[0])
            self.layer.built = True
        super(Piecewise2D, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        inputs, rows, cols = inputs
        if K.dtype(rows) != 'int32':
            rows = K.cast(rows, 'int32')
        if K.dtype(cols) != 'int32':
            cols = K.cast(cols, 'int32')
        if self.pos_type == self.POS_TYPE_SEGMENTS:
            rows = K.concatenate([K.zeros((K.shape(inputs)[0], 1), dtype='int32'), rows], axis=1)
            cols = K.concatenate([K.zeros((K.shape(inputs)[0], 1), dtype='int32'), cols], axis=1)
        return K.map_fn(
            lambda i: self._call_sample(inputs, rows, cols, i),
            K.arange(K.shape(inputs)[0]),
            dtype=K.floatx(),
        )

    def _call_sample(self, inputs, rows, cols, index):
        step = 1
        if self.pos_type == self.POS_TYPE_PAIRS:
            step = 2
        inputs, rows, cols = inputs[index], rows[index], cols[index]
        return K.map_fn(
            lambda row: self._call_rows(inputs, rows, cols, row),
            K.arange(1, K.shape(rows)[0], step),
            dtype=K.floatx(),
        )

    def _call_rows(self, inputs, rows, cols, row):
        step = 1
        if self.pos_type == self.POS_TYPE_PAIRS:
            step = 2
        return K.map_fn(
            lambda col: self._call_piece(inputs, rows, cols, row, col),
            K.arange(1, K.shape(cols)[0], step),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, rows, cols, row, col):
        piece = inputs[rows[row - 1]:rows[row], cols[col - 1]:cols[col]]
        return K.squeeze(self.layer.call(inputs=K.expand_dims(piece, axis=0)), axis=0)

    def compute_output_shape(self, input_shape):
        pos_row_len, pos_col_len = input_shape[1][1], input_shape[2][1]
        if self.pos_type == self.POS_TYPE_PAIRS:
            pos_row_len //= 2
            pos_col_len //= 2
        child_output_shape = self.layer.compute_output_shape(input_shape[0])
        return (input_shape[0][0], pos_row_len, pos_col_len) + child_output_shape[1:]

    def compute_mask(self, inputs, mask=None):
        return None
