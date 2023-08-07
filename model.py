from transformers import TFAutoModelForSeq2SeqLM

name = "facebook/bart-large-xsum" # The large xsum version has 1.5GB.

model = TFAutoModelForSeq2SeqLM.from_pretrained(name)
