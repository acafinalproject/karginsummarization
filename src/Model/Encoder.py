import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_rate_1, drop_rate_2, drop_rate_ffn, num_layers):
        super().__init__()
        self.layers_ = tf.keras.models.Sequential()
        for i in range(num_layers):
            self.layers_.add(EncoderLayer(d_model, ffn_hidden, num_heads, drop_rate_1, drop_rate_2, drop_rate_ffn))

    def call(self, x):
        x = self.layers_(x)
        return x


class EncoderLayer(tf.keras.Model):
    "The class performs all functionality in the Encoder Block."

    def __init__(self, d_model, ffn_hidden, num_heads, drop_rate_1, drop_rate_2, drop_rate_ffn):
        super().__init__()
        self.attention = MultiHeadAttention_Encoder(d_model=d_model, num_heads=num_heads)  # Custom Implemented
        self.norm1 = LayerNormalization_Encoder(parameters_shape=[d_model])  # Custom Implemented
        self.dropout1 = tf.keras.layers.Dropout(drop_rate_1)
        self.ffn = PositionwiseFeedForward_Encoder(d_model=d_model, hidden=ffn_hidden,
                                                   drop_rate=drop_rate_ffn)  # Custom Implemented
        self.norm2 = LayerNormalization_Encoder(parameters_shape=[d_model])
        self.dropout2 = tf.keras.layers.Dropout(drop_rate_2)

    def call(self, x):
        residual_x = x  # for skipconnetctions
        x = self.attention(x)  # Attention layer
        x = self.dropout1(x)  # Dropout Layer

        x = self.norm1(x + residual_x)  ## Add and Normalization
        residual_x = x  # for skipconnetctions
        x = self.ffn(x)  # Feed Froward

        x = self.dropout2(x)  ## Dropout
        x = self.norm2(x + residual_x)  ## Add and Normalization
        return x


class MultiHeadAttention_Encoder(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = tf.keras.layers.Dense(units=d_model * 3, use_bias=False)
        self.linear_layer = tf.keras.layers.Dense(d_model)

    def call(self, x):
        batch_size, max_sequence_length, d_model = x.shape
        qkv = self.qkv_layer(x)  # batch_size, #max_sequence_length, #d_model*3
        qkv = tf.reshape(qkv, shape=(batch_size, max_sequence_length, self.num_heads,3 * self.head_dim))  # batch_size x max_sequence_length x num_heads x 3 * self.head_dim
        # We obtain Q, K, and V with a single layer, which is equivalent to breaking it down and constructing Q_i, K_i, and V_i separately for each value of i where num_heads >= i >= 1
        qkv = tf.reshape(qkv, shape=(qkv.shape[0], qkv.shape[2], qkv.shape[1], qkv.shape[3]))
        # Reshape for performing the (Q @ K.t) operation for each head
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=3)
        values, attention = scaled_dot_product_Encoder(q, k, v)
        values = tf.reshape(values, shape=(batch_size, max_sequence_length, self.num_heads * self.head_dim))
        # The above reshape operation is equivalent to concatenation
        out = self.linear_layer(values)
        return out


def scaled_dot_product_Encoder(q, k, v):
    d_k = q.shape[-1]
    scaled = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) / d_k ** 0.5

    attention = tf.keras.activations.softmax(scaled, axis=-1)
    values = tf.matmul(attention, v)
    return values, attention


class PositionwiseFeedForward_Encoder(tf.keras.Model):

    def __init__(self, d_model, hidden, drop_rate=0.1):
        super(PositionwiseFeedForward_Encoder, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden)
        self.linear2 = tf.keras.layers.Dense(d_model, activation="linear")
        self.relu = tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNormalization_Encoder(tf.keras.Model):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = tf.Variable(tf.ones(parameters_shape))
        self.beta = tf.Variable(tf.zeros(parameters_shape))

    def call(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = tf.reduce_mean(inputs, axis=2, keepdims=True)
        var = tf.reduce_mean(((inputs - mean) ** 2), axis=2, keepdims=True)
        std = (var + self.eps) ** 0.5
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
