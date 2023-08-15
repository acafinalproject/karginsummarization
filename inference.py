import os
import tensorflow as tf
from src.components import Transformer, TextTokenizer
from src.components import masked_accuracy, masked_loss
from src.model import TextSummarizer, ExportSummarizer
from dotenv import dotenv_values

from src import logger

config_paths = dotenv_values(".env.paths")
config_structor = dotenv_values(".env.structor")

DATA = config_paths['DATA']
DATA_DIR = config_paths['DATA_DIR']
SAVE_PATH = config_paths['SAVE_PATH']
VOCAB_PATH = config_paths['VOCAB_PATH']
max_length=int(config_structor['max_length'])
d_model = int(config_structor['d_model'])
num_heads = int(config_structor['num_heads'])
dff = int(config_structor['dff'])
num_layers=int(config_structor['num_layers'])

def inference():
    if os.path.exists('summarizer'):
        loaded_model = tf.saved_model.load(export_dir='summarizer')

        summarizer = loaded_model.signatures['serving_default']
    else:
        tokenizer = TextTokenizer(max_length=max_length, vocab_path=VOCAB_PATH)
        vocab_size = tokenizer.get_vocab_size()

        transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
                                dff=dff, input_vocab_size=vocab_size, target_vocab_size=vocab_size)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

    
        transformer.compile(
            loss=masked_loss,
            optimizer=optimizer,
            metrics=[masked_accuracy])

        saved_files = os.listdir(SAVE_PATH)
        
        if len(saved_files) == 0:
            logger.error(f"Saved model is not finded. Please remove {SAVE_PATH} and start training.")

        last_checkpoint = sorted(saved_files)[-1]
        save_path = os.path.join(SAVE_PATH,  last_checkpoint)

        logger.info(f"Import saved model from {save_path}")

        transformer.load_weights(os.path.join(save_path, "cp-transformer.ckpt"))

        summarizer = TextSummarizer(tokenizer, transformer)

        summarizer = ExportSummarizer(summarizer)

        tf.saved_model.save(summarizer, export_dir='summarizer')

    return summarizer

if __name__ == "__main__":
    txt = tf.constant("today i will finish")
    summarizer = inference()

    print(summarizer(txt))
