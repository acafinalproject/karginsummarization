from dataset import transformed_data
from transformers import DataCollatorForSeq2Seq


max_input_length = 512
max_target_length = 128
prefix = "summarize: "

checkpoint = 't5-small'
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=examples["description"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = transformed_data.map(preprocess_function, batched=True, remove_columns=['title', 'text', 'domain', 'date', 'description', 'url', 'image_url'])


collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
