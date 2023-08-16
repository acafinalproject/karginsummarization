from datasets import load_dataset
dataset = load_dataset("cc_news")

max_text_length = 650
def filter_long_texts(example):
    return len(example["text"].split()) < max_text_length

filtered_dataset = dataset["train"].filter(filter_long_texts)
splited_data = filtered_dataset.train_test_split(test_size=0.1)

test_valid = splited_data['test'].train_test_split(test_size=0.5)

transformed_data = DatasetDict({
    'train': splited_data['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

transformed_data["train"] = transformed_data["train"].shuffle(seed=123).select(range(70000))
transformed_data["test"] = transformed_data["test"].shuffle(seed=123).select(range(6500))
transformed_data["validation"] = transformed_data["test"].shuffle(seed=123).select(range(6500))