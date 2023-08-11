import tensorflow as tf
import tensorflow_datasets as tfds
from src.utils import Transformer
from transformers import AutoTokenizer

article = """australian shares closed"""


# input_ids = tokenizer(article, return_tensors="tf").input_ids

class Summarizer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, max_target_len=50, dropout_rate=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_gigaword", cache_dir="home/samvel/aca/data")
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.max_target_len = max_target_len

        self.eos = self.tokenizer.eos_token_id
        self.bos = self.tokenizer.bos_token_id

        self.input_vocab_size = self.tokenizer.vocab_size
        self.target_vocab_size = self.tokenizer.vocab_size

        self.model = Transformer(num_layers=self.num_layers, d_model=self.d_model, num_heads=self.num_heads, dff=self.dff, 
                                 input_vocab_size=self.input_vocab_size, target_vocab_size=self.target_vocab_size, dropout_rate=self.dropout_rate)

    @tf.function
    def call(self, x):
        x_tokenized = self.tokenizer(x, return_tensors="tf").input_ids
        
        current_output = None
        target_len = 0

        output = [self.bos]

        while current_output != self.eos and target_len <= self.max_target_len:
            current_output = self.model((x_tokenized, output))

            assert type(current_output) == int | "Current output is not int"

            output.append(current_output)

        return self.tokenizer.decode(output)

    
    def train_step(self, ):
        pass

    @property
    def metrics():
        pass
