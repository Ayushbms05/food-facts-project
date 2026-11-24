
import random
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42

MODEL_NAME = "distilbert/distilbert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 3
LEARNING_RATE = 2e-5
PATIENCE = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "deep_learning"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = DATA_DIR / "distilbert_food_nova"

LABEL_NAMES = ["NOVA 1", "NOVA 2", "NOVA 3", "NOVA 4"]


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
        raise ValueError(f"{name} data contains missing ingredient text.")

    if df["nova_group"].isna().any():
        raise ValueError(f"{name} data contains missing NOVA labels.")

    df["ingredient_text"] = df["ingredient_text"].astype(str)
    df["nova_group"] = df["nova_group"].astype(int)

    if not df["nova_group"].isin([1, 2, 3, 4]).all():
        raise ValueError(f"{name} data contains invalid NOVA labels.")

print(f"Training products: {len(train_df):,}")
print(f"Validation products: {len(validation_df):,}")
print(f"Testing products: {len(test_df):,}")


# --------------------------------------------------
# 4. Load pretrained tokenizer
# --------------------------------------------------

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")


# --------------------------------------------------
# 5. Prepare PyTorch dataset
# --------------------------------------------------

class FoodTextDataset(Dataset):

    def __init__(self, dataframe):

        self.texts = dataframe["ingredient_text"].tolist()

        # Convert NOVA 1-4 into class indices 0-3.
        self.labels = torch.tensor(
            dataframe["nova_group"].to_numpy(dtype=np.int64) - 1,
            dtype=torch.long,
        )

        # Tokenize all text once instead of repeating each epoch.
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


print("\nTokenizing datasets...")

train_dataset = FoodTextDataset(train_df)
validation_dataset = FoodTextDataset(validation_df)
test_dataset = FoodTextDataset(test_df)

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

print("Tokenization completed.")


# --------------------------------------------------
# 6. Load pretrained DistilBERT classifier
# --------------------------------------------------

print("\nLoading pretrained DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
)

model.to(DEVICE)

print("DistilBERT model loaded successfully.")


# --------------------------------------------------
# 7. Optimizer and loss
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 8. Evaluation function
# --------------------------------------------------

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(outputs.logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
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

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions,
    )


# --------------------------------------------------
# 9. Training loop with gradient accumulation
# --------------------------------------------------

best_validation_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_train_loss = 0.0
    all_train_predictions = []
    all_train_labels = []

    optimizer.zero_grad()

    print(f"\nStarting epoch {epoch}/{EPOCHS}")

    for batch_index, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(outputs.logits, labels)

        # Scale loss for gradient accumulation.
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()

        # Correctly handle the final incomplete accumulation group.
        is_last_batch = batch_index == len(train_loader) - 1

        if (
            (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or is_last_batch
        ):

            if is_last_batch:
                remaining = (
                    (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS
                )

                if remaining:
                    correction = (
                        GRADIENT_ACCUMULATION_STEPS / remaining
                    )

                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            optimizer.zero_grad()

        total_train_loss += loss.item() * labels.size(0)

        predictions = outputs.logits.argmax(dim=1)

        all_train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_train_labels.extend(
            labels.detach().cpu().numpy()
        )

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed {batch_index + 1}/{len(train_loader)} batches"
            )

    train_loss = total_train_loss / len(train_dataset)

    train_accuracy = accuracy_score(
        all_train_labels,
        all_train_predictions,
    )

    train_f1 = f1_score(
        all_train_labels,
        all_train_predictions,
        average="macro",
    )

    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _,
    ) = evaluate(validation_loader)

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

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        print("Saved new best model and tokenizer.")

    else:

        epochs_without_improvement += 1
        print("No validation improvement.")

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# 10. Load best model
# --------------------------------------------------

print("\nLoading best saved checkpoint...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
)

model.to(DEVICE)


# --------------------------------------------------
# 11. Final test evaluation
# --------------------------------------------------

(
    test_loss,
    test_accuracy,
    test_macro_f1,
    test_labels,
    test_predictions,
) = evaluate(test_loader)


print("\n" + "=" * 55)
print("FINAL DISTILBERT TEST RESULTS")
print("=" * 55)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro F1: {test_macro_f1:.4f}")

print("\nClassification report:")

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1, 2, 3],
        target_names=LABEL_NAMES,
        zero_division=0,
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1, 2, 3],
    )
)

print(f"\nBest model and tokenizer saved in: {MODEL_DIR}")
