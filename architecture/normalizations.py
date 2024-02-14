import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


"""ADAPTIVE INSTANCE NORMALIZATION"""
class AdaINNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(AdaINNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Due set di pesi per la scala e l'offset
        shape = (input_shape[-1],)
        # parametro usato per regolare l'ampiezza dei valori in uscita dalla normalizzazione
        self.scale = self.add_weight(name='scale', shape=shape, initializer='ones', trainable=True)
        # parametro usato per traslare i valori normalizzati lungo l'asse dell'ampiezza
        self.offset = self.add_weight(name='offset', shape=shape, initializer='zeros', trainable=True)

    def call(self, x):
        # Calcola media e deviazione standard di x
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv

        # Applica scale e offset
        return self.scale * normalized + self.offset
