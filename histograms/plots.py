from data_tokenizer import dataset, tokenizer
import matplotlib.pyplot as plt

doc_len_train = [len(tokenizer.encode(sample)) for sample in dataset['train']['document']]
summary_len_train = [len(tokenizer.encode(sample)) for sample in dataset['train']['summary']]

fix, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

axes[0].hist(doc_len_train, bins=20, color="C0", edgecolor="C0")
axes[0].set_title("Document token length (Train)")
axes[0].set_xlabel("Length")
axes[0].set_ylabel("Count")

axes[1].hist(summary_len_train, bins=20, color="C0", edgecolor="C0")
axes[1].set_title("Summary token length (Train)")
axes[1].set_xlabel("Length")

plt.tight_layout()
plt.show()
