import tensorflow as tf
from dotenv import dotenv_values

config_training=dotenv_values(".env.training")

BUFFER_SIZE = int(config_training['BUFFER_SIZE'])
BATCH_SIZE = int(config_training['BATCH_SIZE'])



class TextTokenizer(tf.keras.layers.Layer):
    def __init__(self, max_length, vocab_path):
        super().__init__()
        self.max_length = max_length
        self.vocab_path = vocab_path

        self.vectorizer = tf.keras.layers.TextVectorization(output_sequence_length=max_length, 
                                                            standardize=None, vocabulary=vocab_path)
        
        self.vocabulary = self.vectorizer.get_vocabulary()

        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"

    def call(self, inputs):
        inputs = tf.strings.join([self.bos_token, inputs, self.eos_token], separator=" ")
        tok_inputs = self.vectorizer(inputs)

        return tok_inputs

    def make_batches(self, ds):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))
            
    def prepare_batch(self, doc, sum):
        MAX_TOKENS = self.max_length

        doc = self(doc)      # Output is ragged.
        doc = doc[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.

        sum = self(sum)
        sum = sum[:, :(MAX_TOKENS+1)]
        sum_inputs = sum[:, :-1]  # Drop the [END] tokens
        sum_labels = sum[:, 1:]   # Drop the [START] tokens

        return (doc, sum_inputs), sum_labels

    def get_vocabulary(self):
        return self.vocabulary

    def detokenize(self, output):
        # decoded_words = tf.gather(self.vocabulary, output)
        # result = tf.strings.join(decoded_words, separator=" ")
        return tf.gather(self.vocabulary, output)

    def get_vocab_size(self):
        return len(self.vectorizer.get_vocabulary()) 

    def get_config(self):
        return {
            "max_length": self.max_length
        }


