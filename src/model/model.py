from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

def collator(tokenizer, checkpoint):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
def model(checkpoint):
    return AutoModelForSeq2SeqLM.from_pretrained(checkpoint)