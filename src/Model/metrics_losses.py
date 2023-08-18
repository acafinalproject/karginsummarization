import tensorflow as tf
## Loss function code Start

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

## Loss function code End

## Positional Encodeing Code Start

class PositionalEncoding(tf.keras.Model):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def call(self):

        even_i = tf.cast(tf.range(0,self.d_model,2),tf.float32)
        denominator = tf.pow(10000.0, even_i/self.d_model)

        position = tf.reshape(tf.range(self.max_sequence_length),shape=(self.max_sequence_length, 1))
        position = tf.cast(position,tf.float32)
        even_PE = tf.sin(position / denominator)
        odd_PE = tf.cos(position / denominator)
        stacked = tf.stack([even_PE, odd_PE], axis=2)
        PE = tf.reshape(stacked,shape=(self.max_sequence_length,self.d_model))
        return PE

## Positional Encodeing Code End

## LearningRateSchedule Code Start

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

## LearningRateSchedule Code End


##Custom Accuracy CODE Start
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='masked_accuracy', **kwargs):
        super(MaskedAccuracy, self).__init__(name=name, **kwargs)
        self.matched_count = self.add_weight(name='matched_count', initializer='zeros')
        self.mask_count = self.add_weight(name='mask_count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=2)
        y_true = tf.cast(y_true, y_pred.dtype)
        match = y_true == y_pred

        mask = y_true != 0

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        self.matched_count.assign_add(tf.reduce_sum(match))
        self.mask_count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.matched_count / self.mask_count

    def reset_state(self):
        self.matched_count.assign(0.0)
        self.mask_count.assign(0.0)

##Custom Accuracy CODE Start