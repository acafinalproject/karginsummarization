import tensorflow as tf
import tensorflow_datasets as tfds
from src import Summarizer

data, info = tfds.load('gigaword',
            data_dir='/home/samvel/aca/data',
            with_info=True,
            split='train',
            as_supervised=True)



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_gigaword", cache_dir="home/samvel/aca/data")

phrase = ["Hop hey jan jan", "I hope this will work"]

print(tokenizer(phrase))


# SHOW EXAMPLES OF "GIGAWORD"
# i = 0

# for document, summary in data:
#     print("____________________________")
#     print(document)
#     print(summary)
#     print(len(document.numpy().decode('ascii').split()), " - ", len(summary.numpy().decode('ascii').split()))
#     print("____________________________")

#     if i > 30:
#         break
#     i += 1