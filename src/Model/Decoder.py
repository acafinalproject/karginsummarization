import tensorflow as tf
from Attenction_and_ffd import Decoder_attention,CrossAttention,FeedForward
class SequentialDecoder(tf.keras.Sequential):
    def call(self, inputs):
        target, context = inputs
        for layer in self.layers:
            target = layer((target, context))
        return target

class DecoderLayer(tf.keras.Model):
    def __init__(self,d_model,num_heads,dff,value_dim,dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.Decoder_attention = Decoder_attention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            value_dim = value_dim,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            value_dim = value_dim,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, target_context):
        target,context = target_context
        target= self.Decoder_attention(target)
        x = self.cross_attention((target,context))
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, value_dim, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = SequentialDecoder()
        for i in range(num_heads):
            self.dec_layers.add(DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate,
                                             value_dim=value_dim))

        self.last_attn_scores = None

    def call(self, x_context):
        x, context = x_context
        x = self.dropout(x)
        x = self.dec_layers((x, context))
        self.last_attn_scores = self.dec_layers.layers[-1].last_attn_scores
        return x