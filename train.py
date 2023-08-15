import os
import tensorflow as tf
from src.components import Transformer, TextTokenizer, CustomCallback
from src.components import prepare_dataset, masked_loss, masked_accuracy
from src.model import TextSummarizer, ExportSummarizer

from src import logger
from dotenv import dotenv_values

config_path=dotenv_values(".env.paths")
config_structor=dotenv_values(".env.structor")
config_training=dotenv_values(".env.training")

DATA=config_path['DATA']
DATA_DIR=config_path['DATA_DIR']
SAVE_PATH=config_path['SAVE_PATH']
VOCAB_PATH=config_path['VOCAB_PATH']
max_length=int(config_structor['max_length'])
d_model=int(config_structor['d_model'])
num_heads=int(config_structor['num_heads'])
dff=int(config_structor['dff'])
num_layers=int(config_structor['num_layers'])
epochs=int(config_training['epochs'])
every_n_batch=int(config_training['every_n_batch'])

def train():
    data = prepare_dataset(DATA, DATA_DIR)

    train_examples, val_examples = data['train'], data['test']

    tokenizer = TextTokenizer(max_length=max_length, vocab_path=VOCAB_PATH)
    vocab_size = tokenizer.get_vocab_size()

    train_batches = tokenizer.make_batches(train_examples)
    val_batches = tokenizer.make_batches(val_examples)

    transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
                            dff=dff, input_vocab_size=vocab_size, target_vocab_size=vocab_size)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])
    
    cp_callback = CustomCallback(SAVE_PATH, every_n_batch=every_n_batch)

    saved_files = os.listdir(SAVE_PATH)
    
    if saved_files:
        last_checkpoint = sorted(saved_files)[-1]

        save_path = os.path.join(SAVE_PATH, last_checkpoint)
        logger.info(f"Import saved model from {save_path}")

        transformer.load_weights(os.path.join(save_path, "cp-transformer.ckpt"))


    transformer.fit(train_batches,
                epochs=epochs,
                validation_data=val_batches,
                callbacks=[cp_callback])
    

if __name__ == '__main__':
    train()


# import os
# os.makedirs('drive/MyDrive/project_folder', exist_ok=True)
# %cd drive/MyDrive/project_folder
# !git clone https://github.com/acafinalproject/textsummarization.git
# %cd textsummarization
# !git checkout training_example
# !pip install python-dotenv
# !python train.py