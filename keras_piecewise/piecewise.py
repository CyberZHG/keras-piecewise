import keras
import keras.backend as K


class Piecewise(keras.layers.Wrapper):

    def __init__(self,
                 layer,
                 **kwargs):
        """Initialize the wrapper layer.

        :param layer: The layer that process the inner pieces.
        :param piece_num: The fixed number of pieces for each data.
        :param kwargs: Arguments for parent.
        """
        self.layer = layer
        self.supports_masking = True
        super(Piecewise, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.input_spec = list(map(lambda x: keras.engine.InputSpec(shape=x), input_shape))
        child_input_shape = (None,) + input_shape[0][1:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(Piecewise, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        inputs, positions = inputs
        positions = K.concatenate([K.zeros((K.shape(inputs)[0], 1), dtype='int32'), positions], axis=1)
        print(K.map_fn(
            lambda i: self._call_sample(inputs, positions, i),
            K.arange(K.shape(inputs)[0]),
            dtype=K.floatx(),
        ))
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
            K.arange(1, K.shape(positions)[0]),
            dtype=K.floatx(),
        )

    def _call_piece(self, inputs, positions, index):
        piece = inputs[positions[index - 1]:positions[index]]
        return K.squeeze(self.layer.call(inputs=K.expand_dims(piece, axis=0)), axis=0)

    def compute_output_shape(self, input_shape):
        child_input_shape = (None,) + input_shape[0][1:]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        return (input_shape[0][0], input_shape[1][1]) + child_output_shape[1:]

    def compute_mask(self, inputs, mask=None):
        return None
