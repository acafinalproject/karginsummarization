import tensorflow as tf

# TODO: ADD LOGS

@tf.function
def positional_encoding(length, depth):
    depth = depth / 2

    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]     # (seq, 1)
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = tf.concat(
        [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
        axis=-1) 

    return pos_encoding

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    @tf.function
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x