import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset('multi_news')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-xsum')

#print(dataset)

#train_ds = dataset['train']
#train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']
#print(type(test_ds))
#print(d, 'document')
#print(s, 'summary'))
