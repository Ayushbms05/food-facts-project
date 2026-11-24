
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42

MAX_VOCAB_SIZE = 30000
MIN_WORD_FREQUENCY = 2
MAX_LENGTH = 256

EMBEDDING_DIM = 128
HIDDEN_SIZE = 128

BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001
PATIENCE = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "deep_learning"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = DATA_DIR / "lstm_best.pt"
VOCAB_PATH = DATA_DIR / "lstm_vocab.json"


# --------------------------------------------------
# 2. Reproducibility and device
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

for name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df),
]:
    if df["ingredient_text"].isna().any():
        raise ValueError(f"{name} dataset contains missing ingredient text.")

    if df["nova_group"].isna().any():
        raise ValueError(f"{name} dataset contains missing NOVA labels.")

    df["ingredient_text"] = df["ingredient_text"].astype(str)
    df["nova_group"] = df["nova_group"].astype(int)

print(f"Training products: {len(train_df):,}")
print(f"Validation products: {len(validation_df):,}")
print(f"Testing products: {len(test_df):,}")


# --------------------------------------------------
# 4. Tokenization and vocabulary
# --------------------------------------------------

TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


# Build vocabulary using training data only.
word_counts = Counter()

for text in train_df["ingredient_text"]:
    word_counts.update(tokenize(text))

# 0 = padding, 1 = unknown word
vocab = {"<PAD>": 0, "<UNK>": 1}

for word, count in word_counts.most_common():
    if count < MIN_WORD_FREQUENCY:
        break

    if len(vocab) >= MAX_VOCAB_SIZE:
        break

    vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab):,}")


def encode_text(text):
    token_ids = [vocab.get(word, 1) for word in tokenize(text)]

    token_ids = token_ids[:MAX_LENGTH]
    actual_length = len(token_ids)

    # Empty text is represented by one unknown token.
    if actual_length == 0:
        token_ids = [1]
        actual_length = 1

    token_ids += [0] * (MAX_LENGTH - len(token_ids))

    return token_ids, actual_length


# --------------------------------------------------
# 5. PyTorch Dataset
# --------------------------------------------------

class IngredientDataset(Dataset):

    def __init__(self, dataframe):
        encoded = [
            encode_text(text)
            for text in dataframe["ingredient_text"]
        ]

        self.texts = torch.tensor(
            [item[0] for item in encoded],
            dtype=torch.long,
        )

        self.lengths = torch.tensor(
            [item[1] for item in encoded],
            dtype=torch.long,
        )

        # NOVA 1-4 becomes class indices 0-3.
        self.labels = torch.tensor(
            dataframe["nova_group"].to_numpy(dtype=np.int64) - 1,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.texts[index],
            self.lengths[index],
            self.labels[index],
        )


train_dataset = IngredientDataset(train_df)
validation_dataset = IngredientDataset(validation_df)
test_dataset = IngredientDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# 6. Define the Bidirectional LSTM
# --------------------------------------------------

class FoodLSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0,
        )

        # Forward direction
        self.forward_lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        # Backward direction
        self.backward_lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.4)

        self.classifier = nn.Linear(
            HIDDEN_SIZE * 2,
            4,
        )

    def forward(self, input_ids, lengths):

        # Convert word IDs into embeddings.
        embedded = self.embedding(input_ids)

        batch_size, sequence_length, _ = embedded.shape

        # Forward LSTM processes words from beginning to end.
        forward_output, _ = self.forward_lstm(embedded)

        # Reverse only the real tokens in each sequence.
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        reverse_positions = torch.where(
            positions < lengths.unsqueeze(1),
            lengths.unsqueeze(1) - 1 - positions,
            positions,
        )

        reversed_embedded = embedded.gather(
            1,
            reverse_positions.unsqueeze(-1).expand(
                -1, -1, EMBEDDING_DIM
            ),
        )

        # Backward LSTM processes reversed valid tokens.
        backward_output, _ = self.backward_lstm(reversed_embedded)

        # Select the last valid output from each direction.
        batch_indices = torch.arange(
            batch_size,
            device=input_ids.device,
        )

        last_indices = lengths - 1

        forward_features = forward_output[
            batch_indices, last_indices
        ]

        backward_features = backward_output[
            batch_indices, last_indices
        ]

        # Combine both directional representations.
        combined = torch.cat(
            [forward_features, backward_features],
            dim=1,
        )

        combined = self.dropout(combined)

        return self.classifier(combined)


model = FoodLSTM(len(vocab)).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 7. Training and evaluation
# --------------------------------------------------

def run_epoch(loader, training):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for input_ids, lengths, labels in loader:

        input_ids = input_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(input_ids, lengths)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# 8. Train with early stopping
# --------------------------------------------------

best_validation_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):

    train_loss, train_accuracy, train_f1 = run_epoch(
        train_loader,
        training=True,
    )

    val_loss, val_accuracy, val_f1 = run_epoch(
        validation_loader,
        training=False,
    )

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Train macro F1: {train_f1:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_f1:.4f}")

    if val_f1 > best_validation_f1:

        best_validation_f1 = val_f1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), MODEL_PATH)

        print("Saved new best model.")

    else:
        epochs_without_improvement += 1
        print("No validation improvement.")

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# 9. Save vocabulary and reload best model
# --------------------------------------------------

with open(VOCAB_PATH, "w", encoding="utf-8") as file:
    json.dump(vocab, file, ensure_ascii=False)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

print(f"\nBest model saved to: {MODEL_PATH}")
print(f"Vocabulary saved to: {VOCAB_PATH}")


# --------------------------------------------------
# 10. Final test evaluation
# --------------------------------------------------

_, test_accuracy, test_macro_f1 = run_epoch(
    test_loader,
    training=False,
)

all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():

    for input_ids, lengths, labels in test_loader:

        input_ids = input_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)

        outputs = model(input_ids, lengths)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(labels.numpy())


print("\n" + "=" * 50)
print("FINAL LSTM TEST RESULTS")
print("=" * 50)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro F1: {test_macro_f1:.4f}")

print("\nClassification report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            "NOVA 1",
            "NOVA 2",
            "NOVA 3",
            "NOVA 4",
        ],
        zero_division=0,
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
    )
)
