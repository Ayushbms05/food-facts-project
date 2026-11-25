
from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import Dataset, DataLoader
from src.deep_learning.inference.predict import load_model


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "deep_learning" / "reports"

INPUT_FILE = PROJECT_ROOT / "data" / "deep_learning" / "datasets" / "natural_distribution_sample.csv"

BATCH_SIZE = 16
MAX_LENGTH = 256

LABEL_NAMES = [
    "NOVA 1",
    "NOVA 2",
    "NOVA 3",
    "NOVA 4",
]


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

class FoodDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.texts = dataframe["ingredient_text"].tolist()

        self.labels = torch.tensor(
            dataframe["nova_group"].to_numpy(dtype="int64") - 1,
            dtype=torch.long,
        )

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


# --------------------------------------------------
# 3. Main evaluation
# --------------------------------------------------

def main():

    print("Loading natural-distribution sample...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The evaluation dataset is empty.")

    if df["ingredient_text"].isna().any():
        raise ValueError("Missing ingredient text found.")

    if df["nova_group"].isna().any():
        raise ValueError("Missing NOVA labels found.")

    df["nova_group"] = df["nova_group"].astype(int)

    if not df["nova_group"].isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid NOVA labels found.")

    print("Products loaded:", len(df))

    print("\nLoading saved DistilBERT model...")

    tokenizer, model = load_model()

    dataset = FoodDataset(df, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("\nStarting evaluation...")

    model.eval()

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].numpy()
            )

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{(batch_index + 1) * BATCH_SIZE:,} "
                    f"products"
                )

    # --------------------------------------------------
    # 4. Calculate metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    correct = sum(
        actual == predicted
        for actual, predicted in zip(
            all_labels,
            all_predictions,
        )
    )

    print("\n" + "=" * 60)
    print("NATURAL-DISTRIBUTION EVALUATION RESULTS")
    print("=" * 60)

    print(f"Total products: {len(all_labels):,}")
    print(f"Correct predictions: {correct:,}")
    print(f"Incorrect predictions: {len(all_labels) - correct:,}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2, 3],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            all_labels,
            all_predictions,
            labels=[0, 1, 2, 3],
        )
    )

    # --------------------------------------------------
    # 5. Save individual predictions
    # --------------------------------------------------

    df["predicted_nova"] = [
        prediction + 1
        for prediction in all_predictions
    ]

    df["correct"] = (
        df["nova_group"] == df["predicted_nova"]
    )

    output_file = PROJECT_ROOT / "data" / "deep_learning" / "reports" / "natural_distribution_predictions.csv"

    df.to_csv(output_file, index=False)

    print("\nPredictions saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
