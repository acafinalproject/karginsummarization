import tensorflow as tf
import numpy as np
class SequentialDecoder(tf.keras.Sequential):
    def call(self, inputs):
        # X shape = bathc_size x max_seq_lenght_x, D_model
        # Y shape = bathc_size x max_seq_lenght_y, D_model
        # mask shape = max_seq_lenght_y x max_seq_lenght_y
        x, y, mask,inference = inputs
        for indx, layer in enumerate(self.layers):
            y = layer((x, y, mask,inference))
        return y


class Decoder(tf.keras.Model):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_rate_1, drop_rate_2, drop_rate_3, drop_rate_ffn,
                 num_layers):
        super().__init__()
        self.model_ = SequentialDecoder()
        for i in range(num_layers):
            self.model_.add(
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_rate_1, drop_rate_2, drop_rate_3, drop_rate_ffn))

    def call(self, inputs,):
        x, y, mask,inference = inputs
        y = self.model_((x, y, mask,inference))
        return y


def scaled_dot_product_Decoder(q, k, v, mask,inference):
    d_k = q.shape[-1]
    scaled = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) / d_k ** 0.5
    if inference:
        attention = tf.keras.activations.softmax(scaled, axis=-1)
        values = tf.matmul(attention, v)
        return values, attention

    if mask is not None:
        scaled += mask

    attention = tf.keras.activations.softmax(scaled, axis=-1)
    values = tf.matmul(attention, v)
    return values, attention




class LayerNormalization_Decoder(tf.keras.Model):
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


class DecoderLayer(tf.keras.Model):
    "The class performs all functionality in the Decoder Block."

    def __init__(self, d_model, ffn_hidden, num_heads, drop_rate_1, drop_rate_2, drop_rate_3, drop_rate_ffn):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention_Decoder(d_model=d_model, num_heads=num_heads)  # Custom Implemented

        self.norm1 = LayerNormalization_Decoder(parameters_shape=[d_model])  # Custom Implemented
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate_1)
        self.encoder_decoder_attention = MultiHeadCrossAttention_Decoder(d_model=d_model,
                                                                         num_heads=num_heads)  # Custom Implemented

        self.norm2 = LayerNormalization_Decoder(parameters_shape=[d_model])
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate_2)
        self.ffn = PositionwiseFeedForward_Decoder(d_model=d_model, hidden=ffn_hidden,
                                                   drop_rate=drop_rate_ffn)  # Custom Implemented
        self.norm3 = LayerNormalization_Decoder(parameters_shape=[d_model])
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate_3)

    def call(self, inputs):
        x, y, mask,inference = inputs
        _y = y
        y = self.self_attention(y,inference=inference, mask=mask)  # Masked_Multihed_Attention_Lyaer
        y = self.norm1(y + _y)  # Add & Normalizetion
        _y = y  # For_Skip_connections

        y = self.encoder_decoder_attention(x, y)  # Encoder_And_Decoder_Attention
        y = self.dropout2(y)  # Drpoout
        y = self.norm2(y + _y)  # #Add & Normalizetion
        _y = y  # For_Skip_connections

        y = self.ffn(y)  # Feed Froward
        y = self.dropout3(y)  # Dropout
        y = self.norm3(y + _y)  # Add & Normalizetion
        return y


class MultiHeadCrossAttention_Decoder(tf.keras.Model):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = tf.keras.layers.Dense(d_model * 2, use_bias=True)  # For Key and Value Layer
        self.q_layer = tf.keras.layers.Dense(d_model, use_bias=True)  # For Query layer
        self.linear_layer = tf.keras.layers.Dense(d_model, use_bias=True)  # feed forward layer

    def call(self, x, y):
        batch_size_x, sequence_length_x, d_model_x = x.shape
        batch_size_y, sequence_length_y, d_model_y = y.shape

        kv = self.kv_layer(x)  # generating Keys and Values
        q = self.q_layer(y)  # genearing Querys

        kv = tf.reshape(kv, shape=(batch_size_x, sequence_length_x, self.num_heads, 2 * self.head_dim))
        q = tf.reshape(q, shape=(batch_size_y, sequence_length_y, self.num_heads, self.head_dim))
        kv = tf.reshape(kv, shape=(kv.shape[0], kv.shape[2], kv.shape[1],
                                   kv.shape[3]))  # Reshape for performing the (Q @ K.t) operation for each head

        q = tf.reshape(q, shape=(
        q.shape[0], q.shape[2], q.shape[1], q.shape[3]))  # Reshape for performing the (Q @ K.t) operation for each head

        k, v = tf.split(value=kv, num_or_size_splits=2,
                        axis=3)  # K: batch _size x num_heads x max_seq_lenght x d_kq, v: batch _size x num_heads x max_seq_lenght x d_kq
        values, attention = scaled_dot_product_Decoder(q, k, v,
                                                       mask=None,inference=False)  # batch _size x num_heads x max_seq_lenght x d_kq
        values = tf.reshape(values, shape=(batch_size_y, sequence_length_y, d_model_y))
        # The above reshape operation is equivalent to concatenation

        out = self.linear_layer(values)  # batch x max_seq_lenght x d_model
        return out  # batch_size x max_seq_lenght x d_model


class MultiHeadAttention_Decoder(tf.keras.Model):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = tf.keras.layers.Dense(d_model * 3, use_bias=True)
        self.linear_layer = tf.keras.layers.Dense(d_model, use_bias=True)

    def call(self, x,inference,mask):
        batch_size, sequence_length, d_model = x.shape
        qkv = self.qkv_layer(x)  # batch_size x max_seq_lenght x d_model*3
        qkv = tf.reshape(qkv, shape=(
            batch_size, sequence_length, self.num_heads,
            3 * self.head_dim))  # batch_size x max_seq_lenght x 8 x 3*d_k_q
        qkv = tf.reshape(qkv, shape=(
            qkv.shape[0], qkv.shape[2], qkv.shape[1], qkv.shape[3]))  # batch_size x 8 x max_seq_lenght x 3*d_k_q
        q, k, v = tf.split(value=qkv, num_or_size_splits=3,
                           axis=3)  # q: batch_size x num_heads x max_seq_lenght x d_k_q, k: batch_size x num_heads x max_seq_lenght x d_k_q, v: batch_size x num_heads x max_seq_lenght x d_k_q
        values, attention = scaled_dot_product_Decoder(q, k, v, mask,inference)  # values: 30 x 8 x 200 x d_k_q
        values = tf.reshape(values, shape=(
            batch_size, sequence_length, self.num_heads * self.head_dim))  # values: 30  x 200 x 512
        out = self.linear_layer(values)
        return out


class PositionwiseFeedForward_Decoder(tf.keras.Model):
    def __init__(self, d_model, hidden, drop_rate=0.1):
        super(PositionwiseFeedForward_Decoder, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.relu = tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
