from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import evaluate
from huggingface_hub import notebook_login

notebook_login()

checkpoint = "t5-small"
dataset = load_dataset("cnn_dailymail", "1.0.0")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

max_text_length = 512
def filter_long_texts(example):
    return len(example["article"].split()) <= max_text_length and len(example["highlights"].split()) >= 10 and len(example["highlights"].split()) <= 128

filtered_train = dataset["train"].filter(filter_long_texts)
filtered_test = dataset["test"].filter(filter_long_texts)
filtered_val = dataset["validation"].filter(filter_long_texts)

transformed_data = DatasetDict({
    'train': filtered_train,
    'test': filtered_test,
    'validation': filtered_val
})

max_input_length = 512
max_target_length = 128
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = transformed_data.map(preprocess_function, batched=True, remove_columns=['article', 'highlights', 'id'])

rouge = evaluate.load("rouge")


