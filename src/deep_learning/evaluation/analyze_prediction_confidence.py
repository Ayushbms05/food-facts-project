
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "reports"
    / "natural_distribution_predictions.csv"
)

df = pd.read_csv(INPUT_FILE)

print("Dataset size:", len(df))

# Identify correct and incorrect predictions.
df["correct"] = df["nova_group"] == df["predicted_nova"]

print("\n========== OVERALL CONFIDENCE ==========")
print(df["confidence"].describe().round(2))

print("\n========== CORRECT PREDICTIONS ==========")
print(
    df.loc[df["correct"], "confidence"]
    .describe()
    .round(2)
)

print("\n========== INCORRECT PREDICTIONS ==========")
print(
    df.loc[~df["correct"], "confidence"]
    .describe()
    .round(2)
)

print("\n========== FALSE NOVA 2 PREDICTIONS ==========")

false_nova2 = df[
    (df["predicted_nova"] == 2)
    & (df["nova_group"] != 2)
]

print("Count:", len(false_nova2))

print(
    false_nova2.groupby("nova_group")["confidence"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\n========== NOVA 4 PREDICTED AS NOVA 3 ==========")

nova4_to_nova3 = df[
    (df["nova_group"] == 4)
    & (df["predicted_nova"] == 3)
]

print("Count:", len(nova4_to_nova3))

print(
    nova4_to_nova3["confidence"]
    .describe()
    .round(2)
)

print("\n========== CONFIDENCE BY PREDICTED CLASS ==========")

print(
    df.groupby("predicted_nova")["confidence"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nConfidence analysis completed.")
