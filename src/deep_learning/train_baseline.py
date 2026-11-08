
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_DIR = Path("data/deep_learning")

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"


# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = {"ingredient_text", "nova_group"}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}")

    df = df.dropna(
        subset=["ingredient_text", "nova_group"]
    ).copy()

    df["ingredient_text"] = df["ingredient_text"].astype(str)
    df["nova_group"] = df["nova_group"].astype(int)

    return df


# --------------------------------------------------
# 3. MAIN
# --------------------------------------------------

def main():

    print("Loading datasets...")

    train_df = load_dataset(TRAIN_PATH)
    validation_df = load_dataset(VALIDATION_PATH)
    test_df = load_dataset(TEST_PATH)

    print(f"Training products: {len(train_df):,}")
    print(f"Validation products: {len(validation_df):,}")
    print(f"Testing products: {len(test_df):,}")

    # --------------------------------------------------
    # 4. BUILD MODEL
    # --------------------------------------------------

    print("\nBuilding TF-IDF + Logistic Regression model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # --------------------------------------------------
    # 5. TRAIN MODEL
    # --------------------------------------------------

    print("\nTraining model...")

    model.fit(
        train_df["ingredient_text"],
        train_df["nova_group"],
    )

    print("Training completed!")

    # --------------------------------------------------
    # 6. VALIDATION RESULTS
    # --------------------------------------------------

    print("\n========== VALIDATION RESULTS ==========")

    validation_predictions = model.predict(
        validation_df["ingredient_text"]
    )

    print(
        "Accuracy:",
        f"{accuracy_score(validation_df['nova_group'], validation_predictions):.4f}",
    )

    print(
        "Macro F1:",
        f"{f1_score(validation_df['nova_group'], validation_predictions, average='macro', zero_division=0):.4f}",
    )

    print("\nClassification Report:")

    print(
        classification_report(
            validation_df["nova_group"],
            validation_predictions,
            labels=[1, 2, 3, 4],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 7. FINAL TEST RESULTS
    # --------------------------------------------------

    print("\n========== FINAL TEST RESULTS ==========")

    test_predictions = model.predict(
        test_df["ingredient_text"]
    )

    print(
        "Accuracy:",
        f"{accuracy_score(test_df['nova_group'], test_predictions):.4f}",
    )

    print(
        "Macro F1:",
        f"{f1_score(test_df['nova_group'], test_predictions, average='macro', zero_division=0):.4f}",
    )

    print("\nClassification Report:")

    print(
        classification_report(
            test_df["nova_group"],
            test_predictions,
            labels=[1, 2, 3, 4],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            test_df["nova_group"],
            test_predictions,
            labels=[1, 2, 3, 4],
        )
    )

    print("\nBaseline experiment complete!")


if __name__ == "__main__":
    main()
