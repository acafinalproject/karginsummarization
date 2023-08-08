from transformers import TFAutoModelForSeq2SeqLM

#name = "facebook/bart-base"
name = "t5-small"

model = TFAutoModelForSeq2SeqLM.from_pretrained(name)
