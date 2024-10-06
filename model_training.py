import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch

#Loading tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)

df = pd.read_csv("Mountain/new_merged_mountain_dataset.csv")
df['tokens'] = df['tokens'].apply(ast.literal_eval)
df['tags'] = df['tags'].apply(ast.literal_eval)

#Splitting Dataset for train and valitaion sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

#Assigning tags to integers
tag2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}

#Align labels with tokens
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    label_index = 0
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        else:
            if label_index < len(labels):
                new_labels.append(tag2id[labels[label_index]])
            if len(new_labels) > 1 and word_id != word_ids[len(new_labels) - 2]:
                label_index += 1
    return new_labels

#Tokenize and align(adjust) labels
def tokenize_and_align_labels(df):
    tokenized_inputs = tokenizer(
        df['tokens'].tolist(),
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        max_length=512
    )
    labels = [
        align_labels_with_tokens(df['tags'].iloc[i], tokenized_inputs.word_ids(batch_index=i))
        for i in range(len(df))
    ]
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

train_tokenized_inputs = tokenize_and_align_labels(train_df)
val_tokenized_inputs = tokenize_and_align_labels(val_df)

#Creating Dataset Class, so PyTorch's DataLoader efficiently load in batches during training and correct formating
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):     #input_ids - tokenized words(token ID)
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = NERDataset(train_tokenized_inputs)
val_dataset = NERDataset(val_tokenized_inputs)

#DataCollator to make sure all sequences in a batch are the same length.
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#Setting up the training arguments
training_args = TrainingArguments(
    output_dir='Mountain/results_new',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='Mountain/logs_new',
    learning_rate=2e-4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

#Trainer setup and begin the training proccess
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()

#Evaluation
outputs = trainer.predict(val_dataset)
predictions = np.argmax(outputs.predictions, axis=2)

#Prepares the model's predictions and true labels
true_labels_flat, predicted_labels_flat = [], []

for i in range(len(val_tokenized_inputs['labels'])):
    true_labels = val_tokenized_inputs['labels'][i]
    predicted_labels_seq = predictions[i]

    for j in range(len(true_labels)):
        if true_labels[j] != -100:  # Ignore special tokens
            true_labels_flat.append(true_labels[j])
            predicted_labels_flat.append(predicted_labels_seq[j])

#Print metrics (Precision, Recall, F1-score)
target_names = ['O', 'B-MOUNTAIN', 'I-MOUNTAIN']
print(classification_report(true_labels_flat, predicted_labels_flat, target_names=target_names))

#Save the model
output_dir = "Mountain/model_new"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model and tokenizer saved successfully.")