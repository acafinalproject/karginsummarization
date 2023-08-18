from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

text = input('Enter text to summarize: ')
text = 'summarize: ' + text
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
inputs = tokenizer(text, return_tensors="pt").input_ids


model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)