
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
MAX_VOCAB_SIZE = 30000
MIN_WORD_FREQUENCY = 2
MAX_LENGTH = 256
EMBEDDING_DIM = 128
CHANNELS = 128
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 0.001
PATIENCE = 3

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "deep_learning" / "datasets"
MODEL_DIR = PROJECT_ROOT / "data" / "deep_learning" / "models"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "cnn_best.pt"
VOCAB_PATH = MODEL_DIR / "cnn_vocab.json"


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


# Build vocabulary ONLY from training data.
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
    tokens = tokenize(text)
    token_ids = [vocab.get(token, 1) for token in tokens]

    # Truncate long descriptions.
    token_ids = token_ids[:MAX_LENGTH]

    # Pad shorter descriptions.
    token_ids += [0] * (MAX_LENGTH - len(token_ids))

    return token_ids


# --------------------------------------------------
# 5. Convert data into PyTorch datasets
# --------------------------------------------------

class IngredientDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = torch.tensor(
            np.array(
                [encode_text(text) for text in dataframe["ingredient_text"]],
                dtype=np.int64,
            )
        )

        # Convert NOVA 1-4 into class indices 0-3.
        self.labels = torch.tensor(
            dataframe["nova_group"].to_numpy(dtype=np.int64) - 1,
            dtype=torch.long,
        )

        self.lengths = torch.tensor(
            [
                min(len(tokenize(text)), MAX_LENGTH)
                for text in dataframe["ingredient_text"]
            ],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.lengths[index], self.labels[index]


train_dataset = IngredientDataset(train_df)
validation_dataset = IngredientDataset(validation_df)
test_dataset = IngredientDataset(test_df)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

validation_loader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# --------------------------------------------------
# 6. Define the 1D CNN
# --------------------------------------------------

class FoodCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0,
        )

        self.convolutions = nn.ModuleList([
            nn.Conv1d(EMBEDDING_DIM, CHANNELS, kernel_size=3, padding=1),
            nn.Conv1d(EMBEDDING_DIM, CHANNELS, kernel_size=5, padding=2),
            nn.Conv1d(EMBEDDING_DIM, CHANNELS, kernel_size=7, padding=3),
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        self.classifier = nn.Linear(CHANNELS * 3, 4)

    def forward(self, input_ids, lengths):
        # Shape: (batch, sequence length, embedding dimension)
        embedded = self.embedding(input_ids)

        # Conv1d expects: (batch, channels, sequence length)
        embedded = embedded.transpose(1, 2)

        pooled_features = []

        for convolution in self.convolutions:
            features = self.relu(convolution(embedded))

            # Ignore positions that are only padding.
            positions = torch.arange(
                features.size(2), device=features.device
            ).unsqueeze(0)

            valid_positions = positions < lengths.unsqueeze(1)

            features = features.masked_fill(
                ~valid_positions.unsqueeze(1), -1e9
            )

            # Global max pooling.
            pooled = torch.max(features, dim=2).values
            pooled_features.append(pooled)

        combined = torch.cat(pooled_features, dim=1)
        combined = self.dropout(combined)

        return self.classifier(combined)


model = FoodCNN(len(vocab)).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 7. Training and evaluation functions
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

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
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
        train_loader, training=True
    )

    val_loss, val_accuracy, val_f1 = run_epoch(
        validation_loader, training=False
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
# 9. Save vocabulary and model settings
# --------------------------------------------------

with open(VOCAB_PATH, "w", encoding="utf-8") as file:
    json.dump(vocab, file, ensure_ascii=False)

# Load the best checkpoint, not simply the last epoch.
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

print(f"\nBest model saved to: {MODEL_PATH}")
print(f"Vocabulary saved to: {VOCAB_PATH}")


# --------------------------------------------------
# 10. Final test evaluation
# --------------------------------------------------

_, test_accuracy, test_macro_f1 = run_epoch(
    test_loader, training=False
)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for input_ids, lengths, labels in test_loader:
        input_ids = input_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)

        outputs = model(input_ids, lengths)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n" + "=" * 50)
print("FINAL CNN TEST RESULTS")
print("=" * 50)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro F1: {test_macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=["NOVA 1", "NOVA 2", "NOVA 3", "NOVA 4"],
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
