import os
import tensorflow as tf
from src.components import Transformer, CustomTokenizer, CustomSchedule
from src.components import masked_accuracy, masked_loss
from src.model import Summarizer, ExportSummarizer
from dotenv import dotenv_values
from PIL import Image
import streamlit as st

from src import logger

config_paths = dotenv_values(".env.paths")
config_structor = dotenv_values(".env.structor")

# get configs
DATA = config_paths['DATA']
DATA_DIR = config_paths['DATA_DIR']
SAVE_PATH = config_paths['SAVE_PATH']
VOCAB_PATH = config_paths['VOCAB_PATH']
d_model = int(config_structor['d_model'])
num_heads = int(config_structor['num_heads'])
dff = int(config_structor['dff'])
num_layers=int(config_structor['num_layers'])

def prepare_summarizer():
    export_dir = os.path.join(DATA_DIR, "summarizer")

    if not os.path.exists(export_dir):
        tokenizer = CustomTokenizer(vocab_path=VOCAB_PATH)
        vocab_size = tokenizer.get_vocab_size()

        transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
                                dff=dff, input_vocab_size=vocab_size, target_vocab_size=vocab_size)
        
        learning_rate = CustomSchedule(d_model)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
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

        summarizer = Summarizer(tokenizer, transformer)

        summarizer = ExportSummarizer(summarizer)

        tf.saved_model.save(summarizer, export_dir=export_dir)

    loaded_model = tf.saved_model.load(export_dir=export_dir)

    summarizer = loaded_model.signatures['serving_default']

    return summarizer

def prepare_output(text):
    gif_path = "/home/samvel/Downloads/processing.gif"

    if text:
        gif_container = st.empty()
        gif_container.image(gif_path, use_column_width=True)

        summarizer = prepare_summarizer()
        tensor_input = tf.constant(text)

        tensor_output = list(summarizer(tensor_input)['output_0'].numpy()[1:-1])

        output = list(map(lambda x: x.decode('utf-8'), tensor_output))
        output = list(filter(lambda x: "UNK" not in x, output))

        gif_container.empty()
        
        summary = " ".join(output)
        
        return summary if summary else "please check that your input is correct"
    
    return ""


def main():
    st.set_page_config(layout="wide")

    st.markdown(
    """
    <style>
    [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
    )

    image = Image.open("aca.png")
    st.image(image)
    st.markdown("<h1 style='text-align: center; color: white;'>Kargin Project</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: gray;'>Kargin Summarization</h2>", unsafe_allow_html=True)

    inp_col, out_col = st.columns(2)

    with inp_col:
        input_text = st.text_area("Enter your text:", "")

    with out_col:
        st.text_area("Summary:", prepare_output(input_text), disabled=True, placeholder="summarization will be here! ;)")


    
    
if __name__ == "__main__":
    main()