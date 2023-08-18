import tensorflow as tf
import Encoder
import Decoder
import torch
from metrics_losses import MaskedAccuracy
from metrics_losses import PositionalEncoding
from metrics_losses import masked_loss
from metrics_losses import CustomSchedule


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_seq_lenght_Y,max_seq_lenght_X, dropout_rate=0.1):
        super().__init__()
        self.max_seq_lenght_Y = max_seq_lenght_Y
        self.accuracy = MaskedAccuracy()
        self.Embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.positional_embeding_encoder = PositionalEncoding(d_model, max_seq_lenght_X)
        self.positional_embeding_decoder = PositionalEncoding(d_model, max_seq_lenght_Y)

        self.encoder = Encoder.Encoder(d_model=d_model, ffn_hidden=dff,
                                       num_heads=num_heads, drop_rate_1=dropout_rate,
                                       drop_rate_2=dropout_rate,
                                       drop_rate_ffn=dropout_rate, num_layers=num_layers)

        self.decoder = Decoder.Decoder(d_model=d_model, ffn_hidden=dff,
                                       num_heads=num_heads, drop_rate_1=dropout_rate,
                                       drop_rate_2=dropout_rate, drop_rate_3=dropout_rate, drop_rate_ffn=dropout_rate,
                                       num_layers=num_layers)

        self.final_layer = tf.keras.layers.Dense(vocab_size, activation="gelu")

    def call(self, inputs):
        x, y = inputs

        x = self.Embedding(x)
        y = self.Embedding(y)
        pos_context = self.positional_embeding_encoder.call()
        pos_target = self.positional_embeding_decoder.call()
        x += pos_context
        y += pos_target
        mask = torch.full([self.max_seq_lenght_Y, self.max_seq_lenght_Y], float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        mask = tf.constant(mask)

        x = self.encoder(x)

        y = self.decoder((x, y, mask, False))
        logits = self.final_layer(y)

        return logits

    def compile(self, loss, optimizer, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.loss = loss
        self.optimizer = optimizer

    def train_step(self, batch):
        context, target, label = batch
        with tf.GradientTape() as tape:
            output = self.call((context, target))
            loss = self.loss(label, output)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.accuracy.update_state(label, output)
        return {"Loss": loss, "train_accuracy": self.accuracy.result()}

    def test_step(self, data):
        # Unpack the data
        context, target, label = data
        # Compute predictions
        y_pred = self.call((context, target))
        self.accuracy.update_state(label, y_pred)
        return {"Validation_accuracy": self.accuracy.result()}

    @property
    def metrics(self):
        return [self.accuracy]

## MOdel construction
Model = Transformer(3,512,8,512,71430,13,34)
learning_rate = CustomSchedule(512)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

Model.compile(
    loss=masked_loss,
    optimizer=optimizer)
