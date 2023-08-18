import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from src.components import prepare_dataset
from dotenv import dotenv_values

bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 29800,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

config_path=dotenv_values(".env.paths")
config_structor=dotenv_values(".env.structor")
config_training=dotenv_values(".env.training")

DATA=config_path['DATA']
DATA_DIR=config_path['DATA_DIR']
SAVE_PATH=config_path['SAVE_PATH']
VOCAB_PATH=config_path['VOCAB_PATH']
MAX_TOKENS_DOC=int(config_structor['max_tokens_doc'])
MAX_TOKENS_SUM=int(config_structor['max_tokens_sum'])
d_model=int(config_structor['d_model'])
num_heads=int(config_structor['num_heads'])
dff=int(config_structor['dff'])
num_layers=int(config_structor['num_layers'])
epochs=int(config_training['epochs'])
every_n_batch=int(config_training['every_n_batch'])

data = prepare_dataset(DATA, DATA_DIR)

train_examples = data['train']

pt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_doc.batch(1000).prefetch(2),
    **bert_vocab_args
)

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)

write_vocab_file('pt_vocab.txt', pt_vocab)

tokenizer = text.BertTokenizer('pt_vocab.txt', **bert_tokenizer_params)

i = 0

for doc1 in train_doc:
    if i == 0:
       doc2 = doc1
    
    i+=1
    if i >= 2:
       break


print(tokenizer.tokenize([doc1.numpy(), doc2.numpy()]).to_tensor())


