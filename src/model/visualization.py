import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import transformed_data

train_text_length = [len(text.split()) for text in transformed_data["train"]["article"]]

print(pd.Series(train_text_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')

plt.hist(train_text_length, bins=90, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('lengths')
plt.ylabel('Counts')
plt.show()

train_s_length = [len(s.split()) for s in transformed_data["train"]["highlights"]]
print(pd.Series(train_s_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(train_s_length, bins=20, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('lengths')
plt.ylabel('Counts')
plt.show()

test_text_length = [len(text.split()) for text in transformed_data["test"]["article"]]
pd.Series(test_text_length).describe()

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(test_text_length, bins=90, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('lengths')
plt.ylabel('Counts')
plt.show()

test_s_length = [len(s.split()) for s in transformed_data["test"]["highlights"]]
print(pd.Series(test_s_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(test_s_length, bins=20, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('lengths')
plt.ylabel('Counts')
plt.show()

val_text_length = [len(text.split()) for text in transformed_data["validation"]["article"]]
print(pd.Series(val_text_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(val_text_length, bins=90, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('lengths')
plt.ylabel('Counts')
plt.show()

val_s_length = [len(s.split()) for s in transformed_data["validation"]["highlights"]]
print(pd.Series(val_s_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(val_s_length, bins=20, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('lengths')
plt.ylabel('Counts')
plt.show()