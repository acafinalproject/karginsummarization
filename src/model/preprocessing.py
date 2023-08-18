# Import required libraries
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import evaluate
from huggingface_hub import notebook_login

# Log in to the Hugging Face Model Hub using notebook_login
notebook_login()

# Define the checkpoint to be used for the model
checkpoint = "t5-small"

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "1.0.0")

# Initialize the tokenizer for the T5 model
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Set maximum length for input text
max_text_length = 512

# Define a function to filter out examples based on length criteria
def filter_long_texts(example):
    return (
        len(example["article"].split()) <= max_text_length and
        len(example["highlights"].split()) >= 10 and
        len(example["highlights"].split()) <= 128
    )

# Apply the filtering function to the training, test, and validation datasets
filtered_train = dataset["train"].filter(filter_long_texts)
filtered_test = dataset["test"].filter(filter_long_texts)
filtered_val = dataset["validation"].filter(filter_long_texts)

# Create a new DatasetDict with the filtered datasets
transformed_data = DatasetDict({
    'train': filtered_train,
    'test': filtered_test,
    'validation': filtered_val
})

# Set maximum lengths for input and target sequences
max_input_length = 512
max_target_length = 128

# Define a prefix to be added to input texts
prefix = "summarize: "

# Define a function to preprocess examples
def preprocess_function(examples):
    # Add the prefix to each input text and tokenize
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokenize target texts and set them as labels
    labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocess_function to the tokenized_data
tokenized_data = transformed_data.map(preprocess_function, batched=True, remove_columns=['article', 'highlights', 'id'])



