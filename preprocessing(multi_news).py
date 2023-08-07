from data_tokenizer import tokenizer, dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(sample):
    document = [text for text in sample['document']]

    document_tokenized = tokenizer(
            document, padding='max_length', truncation=True, max_length=512)
    
    summary = [smr for smr in sample['summary']]
    summary_tokenized = tokenizer(
            summary, padding='max_length', truncation=True, max_length=64)

    document_tokenized['labels'] = summary_tokenized['input_ids']

    return document_tokenized

tokenized_data = dataset.map(preprocess_data, batched=True)
