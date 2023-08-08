from transformers import DataCollatorForSeq2Seq
from data_tokenizer import tokenizer
from model import model
from preprocessing_multi_news import tokenized_data

#bart_base = "facebook/bart-base"
t5 = "t5-small"
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=t5, return_tensors="tf")

from transformers import create_optimizer, AdamWeightDecay

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

tf_train = model.prepare_tf_dataset(
    tokenized_data["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=collator,
)

tf_val = model.prepare_tf_dataset(
        tokenized_data["validation"],
        shuffle=True,
        batch_size=8,
        collate_fn=collator,
)

tf_test = model.prepare_tf_dataset(
    tokenized_data["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=collator,
)


import tensorflow as tf

model.compile(optimizer=optimizer)  # No loss argument!

model.fit(x=tf_train, validation_data=tf_val, epochs=3, batch_size=1)
