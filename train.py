import model
import dataset
import preprocessor
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = model.model()
compute_metrics = model.compute_metrics()
tokenized_data = preprocessor.tokenized_data
collator = preprocessor.tokenized_data

training_args = Seq2SeqTrainingArguments(
    output_dir="",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    gradient_accumulation_steps=10,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    report_to="tensorboard",
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()