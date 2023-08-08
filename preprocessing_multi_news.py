from data_tokenizer import tokenizer, dataset

#def preprocess_data(sample):
#    document = [text for text in sample['document']]
#
#    document_tokenized = tokenizer(
#            document, padding='max_length', truncation=True, max_length=512)
#    
#    summary = [smr for smr in sample['summary']]
#    summary_tokenized = tokenizer(
#            summary, padding='max_length', truncation=True, max_length=64)
#
#    document_tokenized['labels'] = summary_tokenized['input_ids']
#
#    return document_tokenized
#
#tokenized_data = dataset.map(preprocess_data, batched=True)

def preprocess_data(data_to_process):
  #get the dialogue text
  inputs = [dialogue for dialogue in data_to_process["document"]]
  #tokenize text
  model_inputs = tokenizer(inputs,  max_length=512, padding='max_length', truncation=True)

  #tokenize labels
  with tokenizer.as_target_tokenizer():
    targets = tokenizer(data_to_process["summary"], max_length=128, padding='max_length', truncation=True)

  model_inputs['labels'] = targets['input_ids']
  #reuturns input_ids, attention_masks, labels
  return model_inputs

tokenized_data = dataset.map(preprocess_data, batched = True, remove_columns=["document", "summary"])

train_sample = tokenized_data['train'].shuffle(seed=123).select(range(1000))
validation_sample = tokenized_data['validation'].shuffle(seed=123).select(range(500))
test_sample = tokenized_data['test'].shuffle(seed=123).select(range(200))

tokenized_data['train'] = train_sample
tokenized_data['validation'] = validation_sample
tokenized_data['test'] = test_sample
