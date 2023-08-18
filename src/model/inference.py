from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

text = input('Enter text to summarize: ')
text = 'summarize: ' + text
tokenizer = AutoTokenizer.from_pretrained("GagMkrtchyan/T5")
inputs = tokenizer(text, return_tensors="pt").input_ids


model = AutoModelForSeq2SeqLM.from_pretrained("GagMkrtchyan/T5")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
