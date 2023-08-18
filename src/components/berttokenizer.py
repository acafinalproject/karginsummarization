import pathlib
import numpy as np
import re

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

import tensorflow_text as text
import tensorflow as tf

from dotenv import dotenv_values

config_training=dotenv_values(".env.training")
config_structor=dotenv_values(".env.structor")

MAX_TOKENS_DOC=int(config_structor['max_tokens_doc'])
MAX_TOKENS_SUM=int(config_structor['max_tokens_sum'])

BUFFER_SIZE = int(config_training['BUFFER_SIZE'])
BATCH_SIZE = int(config_training['BATCH_SIZE'])

class CustomTokenizer(tf.Module):
  def __init__(self, vocab_path):
    self._reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)

    self.START = tf.argmax(tf.constant(self._reserved_tokens) == "[START]")
    self.END = tf.argmax(tf.constant(self._reserved_tokens) == "[END]")

    self._vocab_path = tf.saved_model.Asset(vocab_path)

    vocab = pathlib.Path(vocab_path).read_text().splitlines()
    self.vocab = tf.Variable(vocab)

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))

    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()

  def make_batches(self, ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(self.prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
          
  def prepare_batch(self, doc, sum):
      doc = self.tokenize(doc)      # Output is ragged.
      doc = doc[:, :MAX_TOKENS_DOC].to_tensor()    # Trim to MAX_TOKENS.

      sum = self.tokenize(sum)
      sum = sum[:, :(MAX_TOKENS_SUM+1)]
      sum_inputs = sum[:, :-1].to_tensor() # Drop the [END] tokens
      sum_labels = sum[:, 1:].to_tensor()   # Drop the [START] tokens

      return (doc, sum_inputs), sum_labels

  def add_start_end(self, ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count,1], self.START)
    ends = tf.fill([count,1], self.END)
    return tf.concat([starts, ragged, ends], axis=1)

  def cleanup_text(self, reserved_tokens, token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result

  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = self.add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return self.cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)
