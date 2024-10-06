import pandas as pd
import re
import ast
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import classification_report

df_mountains = pd.read_csv('Mountain/mountain_dataset_with_markers.csv')

# Function to remove hashtags
df_mountains['text'] = df_mountains['text'].apply(lambda x: re.sub(r'#\w+', '', x))


# Function to split words and included symbols
def custom_tokenize(text):
    return re.findall(r"\w+|[.,!?;]", text)


def generate_bio_labels(sentence, marker):
    words = custom_tokenize(sentence)
    labels = ["O"] * len(words)
    markers = ast.literal_eval(marker)

    for start, end in markers:
        char_index = 0
        for i, word in enumerate(words):
            word_start = char_index
            word_end = char_index + len(word)
            char_index = word_end + 1

            if word_start >= start and word_end <= end:
                if word_start == start:
                    labels[i] = "B-MOUNTAIN"
                else:
                    labels[i] = "I-MOUNTAIN"

    tokenized_text = "['" + "', '".join(words) + "']"
    tokenized_annotation = "['" + "', '".join(labels) + "']"
    return tokenized_text, tokenized_annotation


# Generating BIO labels and formatted tokens
df_mountains[['tokens', 'tags']] = df_mountains.apply(
    lambda row: generate_bio_labels(row['text'], row['markers']),
    axis=1, result_type='expand'
)
df_mountains.drop(columns=['text', 'markers'], inplace=True)

output_csv_filename = "Mountain/data/mountain_new.csv"
df_mountains.to_csv(output_csv_filename, index=False, encoding="utf-8")

print(f"Tokenized dataset with BIO labels saved to {output_csv_filename}")

bio_df = pd.read_csv('Mountain/data/mountain_new.csv')
annotated_df = pd.read_csv('Mountain/annotated_sentences.csv')

bio_df = bio_df.rename(columns={'sentences': 'tokens', 'annotation': 'annotation'})
merged_df = pd.concat([bio_df, annotated_df], ignore_index=True)

output_csv_filename = "Mountain/new_merged_mountain_dataset.csv"
merged_df.to_csv(output_csv_filename, index=False, encoding="utf-8")

print(f"Files successfully merged and saved to {output_csv_filename}")