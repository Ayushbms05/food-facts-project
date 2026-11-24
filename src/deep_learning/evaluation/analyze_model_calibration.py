
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from src.deep_learning.inference.predict import load_model


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "datasets"
    / "validation.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "reports"
)

BATCH_SIZE = 16
MAX_LENGTH = 256
NUM_BINS = 10


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
# 3. Expected Calibration Error
# --------------------------------------------------

def calculate_ece(confidences, correct, num_bins=10):

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    bin_indices = np.digitize(
        confidences,
        bin_edges[1:-1],
        right=False,
    )

    ece = 0.0
    bin_results = []

    for bin_index in range(num_bins):

        mask = bin_indices == bin_index

        count = int(mask.sum())

        if count == 0:
            bin_results.append({
                "bin": bin_index + 1,
                "lower_bound": bin_edges[bin_index],
                "upper_bound": bin_edges[bin_index + 1],
                "count": 0,
                "mean_confidence": np.nan,
                "accuracy": np.nan,
            })
            continue

        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())

        ece += (
            count / len(confidences)
        ) * abs(mean_confidence - accuracy)

        bin_results.append({
            "bin": bin_index + 1,
            "lower_bound": bin_edges[bin_index],
            "upper_bound": bin_edges[bin_index + 1],
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        })

    return ece, pd.DataFrame(bin_results)


# --------------------------------------------------
# 4. Main calibration analysis
# --------------------------------------------------

def main():

    print("Loading validation dataset...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Validation dataset is empty.")

    if df["ingredient_text"].isna().any():
        raise ValueError("Missing ingredient text found.")

    if df["nova_group"].isna().any():
        raise ValueError("Missing NOVA labels found.")

    df["nova_group"] = df["nova_group"].astype(int)

    if not df["nova_group"].isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid NOVA labels found.")

    print("Validation products:", len(df))

    print("\nLoading DistilBERT model...")

    tokenizer, model = load_model()

    dataset = FoodDataset(df, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_probabilities = []
    all_labels = []

    print("\nCalculating validation probabilities...")

    model.eval()

    with torch.no_grad():

        for batch_index, batch in enumerate(loader):

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].numpy()
            )

            if (batch_index + 1) % 20 == 0:

                print(
                    f"Processed "
                    f"{min((batch_index + 1) * BATCH_SIZE, len(df)):,} "
                    f"products"
                )

    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)

    # --------------------------------------------------
    # 5. Calculate calibration metrics
    # --------------------------------------------------

    predictions = probabilities.argmax(axis=1)

    confidences = probabilities.max(axis=1)

    correct = (predictions == labels).astype(float)

    # Expected Calibration Error.
    ece, reliability_df = calculate_ece(
        confidences,
        correct,
        NUM_BINS,
    )

    # Multiclass Brier score.
    one_hot_labels = np.eye(4)[labels]

    brier_score = np.mean(
        np.sum(
            (probabilities - one_hot_labels) ** 2,
            axis=1,
        )
    )

    accuracy = np.mean(correct)

    # --------------------------------------------------
    # 6. Print results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DISTILBERT CALIBRATION RESULTS")
    print("=" * 60)

    print(f"Validation products: {len(labels):,}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Multiclass Brier Score: {brier_score:.4f}")

    print("\nReliability Table:")

    print(
        reliability_df.round(4).to_string(index=False)
    )

    # --------------------------------------------------
    # 7. Save reliability table
    # --------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reliability_file = (
        REPORT_DIR / "calibration_reliability.csv"
    )

    reliability_df.to_csv(
        reliability_file,
        index=False,
    )

    # --------------------------------------------------
    # 8. Generate reliability diagram
    # --------------------------------------------------

    valid_bins = reliability_df[
        reliability_df["count"] > 0
    ]

    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        valid_bins["mean_confidence"],
        valid_bins["accuracy"],
        marker="o",
        label="DistilBERT",
    )

    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("DistilBERT Reliability Diagram")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_file = (
        REPORT_DIR / "calibration_reliability.png"
    )

    plt.savefig(
        graph_file,
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------
    # 9. Save summary
    # --------------------------------------------------

    summary_file = (
        REPORT_DIR / "calibration_summary.txt"
    )

    with open(summary_file, "w") as file:

        file.write("DISTILBERT CALIBRATION RESULTS\n")
        file.write("=" * 40 + "\n")

        file.write(f"Validation products: {len(labels)}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"ECE: {ece:.4f}\n")
        file.write(f"Brier Score: {brier_score:.4f}\n")

    print("\nReports saved:")

    print(reliability_file)
    print(graph_file)
    print(summary_file)

    print("\nCalibration analysis completed successfully.")


if __name__ == "__main__":
    main()
