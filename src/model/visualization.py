import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import transformed_data  # Assuming `transformed_data` is available from a preprocessor module

# Calculate and describe the text lengths of the train dataset
train_text_length = [len(text.split()) for text in transformed_data["train"]["article"]]
print(pd.Series(train_text_length).describe())

# Plot a histogram of train text lengths
plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(train_text_length, bins=90, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('Lengths')
plt.ylabel('Counts')
plt.title('Train Text Lengths')
plt.show()

# Calculate and describe the summary lengths of the train dataset
train_s_length = [len(s.split()) for s in transformed_data["train"]["highlights"]]
print(pd.Series(train_s_length).describe())

# Plot a histogram of train summary lengths
plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(train_s_length, bins=20, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('Lengths')
plt.ylabel('Counts')
plt.title('Train Summary Lengths')
plt.show()

# Similar analysis for the test dataset
test_text_length = [len(text.split()) for text in transformed_data["test"]["article"]]
print(pd.Series(test_text_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(test_text_length, bins=90, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('Lengths')
plt.ylabel('Counts')
plt.title('Test Text Lengths')
plt.show()

test_s_length = [len(s.split()) for s in transformed_data["test"]["highlights"]]
print(pd.Series(test_s_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(test_s_length, bins=20, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('Lengths')
plt.ylabel('Counts')
plt.title('Test Summary Lengths')
plt.show()

# Similar analysis for the validation dataset
val_text_length = [len(text.split()) for text in transformed_data["validation"]["article"]]
print(pd.Series(val_text_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(val_text_length, bins=90, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('Lengths')
plt.ylabel('Counts')
plt.title('Validation Text Lengths')
plt.show()

val_s_length = [len(s.split()) for s in transformed_data["validation"]["highlights"]]
print(pd.Series(val_s_length).describe())

plt.figure(figsize=(14, 7))
plt.style.use('seaborn-whitegrid')
plt.hist(val_s_length, bins=20, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
plt.xlabel('Lengths')
plt.ylabel('Counts')
plt.title('Validation Summary Lengths')
plt.show()
