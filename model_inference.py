import torch
from transformers import BertTokenizerFast, BertForTokenClassification

model_dir = "Mountain/model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForTokenClassification.from_pretrained(model_dir)

# Assigning tags to integers
id2tag = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}


def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    model.eval()  # Set model to evaluation mode and get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze().tolist())  # Convert token IDs to actual tokens

    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()  # Get predicted labels
    predicted_labels = [id2tag[pred] for pred in predictions]
    token_label_pairs = list(zip(tokens, predicted_labels))  # Combine tokens and labels

    # Handling subword tokens
    combined_tokens_labels = []
    current_token = None
    current_label = "O"

    for token, label in token_label_pairs:
        if token.startswith("##"):  # Check if the token is a subword(##)
            if current_token:
                current_token += token[2:]
            else:
                current_token = token[2:]
        else:
            if current_token is not None:  # If it's existing token, append it with his label
                combined_tokens_labels.append((current_token, current_label))
            current_token = token
            current_label = label
    if current_token is not None:  # Last token-label pair if it exists
        combined_tokens_labels.append((current_token, current_label))
    return combined_tokens_labels


def extract_mountain_names(token_label_pairs):
    mountain_names = []
    current_mountain = []

    # mountain names with B-MOUNTAIN and I-MOUNTAIN labels
    for token, label in token_label_pairs:
        if label == "B-MOUNTAIN":
            if current_mountain:
                mountain_names.append(" ".join(current_mountain))
            current_mountain = [token]
        elif label == "I-MOUNTAIN" and current_mountain:
            current_mountain.append(token)
        else:
            if current_mountain:
                mountain_names.append(" ".join(current_mountain))
                current_mountain = []
    if current_mountain:
        mountain_names.append(" ".join(current_mountain))
    return mountain_names


text = input("Enter the text to analyze for mountain name: ")

token_label_pairs = predict(text)

mountain_names = extract_mountain_names(token_label_pairs)

print("\nNamed Entities Recognized as mountains name:")
for mountain_name in mountain_names:
    print(mountain_name)

print("\nTokens and labels predictions:")
for token, label in token_label_pairs:
    print(f'{token}: {label}')