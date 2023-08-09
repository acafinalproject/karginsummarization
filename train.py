from transformers import DataCollatorForSeq2Seq
from data_tokenizer import tokenizer
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback, KerasMetricCallback
from model import model
from datasets import load_metric
from preprocessing_multi_news import tokenized_data
from transformers import create_optimizer, AdamWeightDecay
import os
import numpy as np

import nltk
nltk.download('punkt')


metric = load_metric("rouge")

def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Rouge expects a newline after each sentence
    decoded_predictions = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    result = metric.compute(
        predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return result

#bart_base = "facebook/bart-base"
t5 = "t5-small"
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np")

generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np", pad_to_multiple_of=128)

model_weights_path = "/content/drive/MyDrive/weights/model_weights"
saved_model_path = "/content/drive/MyDrive/model/saved_model"


train_dataset = model.prepare_tf_dataset(
    tokenized_data["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator,
)

validation_dataset = model.prepare_tf_dataset(
    tokenized_data["validation"],
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
)

test_dataset = model.prepare_tf_dataset(
    tokenized_data["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=generation_data_collator
)

push_to_hub_model_id = "t5-small-finetuned-multi_news"

checkpoint_path = "/content/drive/MyDrive/weights/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)  # No loss argument!

push_to_hub_callback = PushToHubCallback(
    output_dir="/content/drive/MyDrive/chkpt/",
    tokenizer=tokenizer,
    hub_model_id=push_to_hub_model_id,
)

metric_callback = KerasMetricCallback(
    metric_fn, eval_dataset=validation_dataset, predict_with_generate=True, use_xla_generation=True
)

callbacks = [metric_callback,  push_to_hub_callback, cp_callback]

model.fit(x=train_dataset, validation_data=validation_dataset, epochs=5, batch_size=4, callbacks=callbacks)


