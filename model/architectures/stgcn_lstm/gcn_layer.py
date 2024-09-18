import tensorflow as tf

class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)

    def call(self, inputs):
        AD, x = inputs
        return tf.einsum("vw,ntwc->ntvc", AD, x)