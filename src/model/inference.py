from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Get input text from the user
text = input('Enter text to summarize: ')

# Prepend "summarize: " to the input text
text = 'summarize: ' + text

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("GagMkrtchyan/T5")

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt").input_ids

# Load the T5 model for sequence-to-sequence tasks
model = AutoModelForSeq2SeqLM.from_pretrained("GagMkrtchyan/T5")

# Generate a summary using the model
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

