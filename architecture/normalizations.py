import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization layer.

        This layer applies normalization to each sample independently,
        calculating mean and variance along the channel axes.
        It introduces two learnable parameters, one for scale and one for offset,
        used to adjust and translate normalized values.

    """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        """Build the layer"""
        # Initialize scale to adjust the amplitude of normalized values
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        # Initialize offset to adjust the translation of normalized values
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        """Forward pass"""
        # Calculate mean and variance across channels
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        # Normalize input using mean and variance
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        # Scale and shift normalized input
        return self.scale * normalized + self.offset