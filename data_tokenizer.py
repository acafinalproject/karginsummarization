from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("multi_news")
tokenizer = AutoTokenizer.from_pretrained("t5-small")


