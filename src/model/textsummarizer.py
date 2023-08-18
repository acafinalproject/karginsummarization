import tensorflow as tf
from dotenv import dotenv_values

config_structor=dotenv_values(".env.structor")

MAX_TOKENS_SUM=int(config_structor['max_tokens_sum'])

class Summarizer(tf.Module):
  def __init__(self, tokenizer, transformer):
    self.tokenizer = tokenizer
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS_SUM):
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    encoder_input = self.tokenizer.tokenize(sentence).to_tensor()

    start_end = self.tokenizer.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = self.tokenizer.detokenize(output)[0]  # Shape: `()`.

    return text

class ExportSummarizer(tf.Module):
  def __init__(self, summarizer):
    self.summarizer = summarizer

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence):
    result = self.summarizer(sentence)

    return result