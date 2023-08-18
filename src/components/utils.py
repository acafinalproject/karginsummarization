import os
from src import logger
from datetime import datetime
import tensorflow as tf
    
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path=None, every_n_batch=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.counter = 0
        self.every_n_batch = every_n_batch

    def on_train_begin(self, logs=None):
        logger.info("Training is started. Good luck! :D")

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        current = "loss-{:.4f}".format(logs.get("loss"))

        if self.counter % self.every_n_batch == 0:
            time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

            save_path = os.path.join(self.save_path, time)
            save_path = '_'.join([save_path, current])

            self.model.save_weights(os.path.join(save_path, "cp-transformer.ckpt"))
            logger.info(f"Model saved at {save_path}")

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


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)
