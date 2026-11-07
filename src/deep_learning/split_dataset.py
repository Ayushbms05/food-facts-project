
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "deep_learning"
INPUT_FILE = DATA_DIR / "training_sample.csv"

RANDOM_STATE = 42
N_CANDIDATES = 500


def normalize_ingredients(df):
    return (
        df["ingredient_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def class_proportions(df, classes):
    return (
        df["nova_group"]
        .value_counts(normalize=True)
        .reindex(classes, fill_value=0)
        .to_numpy()
    )


def find_best_split(df, groups, test_size, target_distribution, seed):
    """
    Try multiple group-based splits and select the one
    with the best size and NOVA class balance.
    """

    splitter = GroupShuffleSplit(
        n_splits=N_CANDIDATES,
        test_size=test_size,
        random_state=seed
    )

    best_score = float("inf")
    best_indices = None

    classes = target_distribution.index

    for train_indices, test_indices in splitter.split(
        df,
        groups=groups
    ):
        candidate_train = df.iloc[train_indices]
        candidate_test = df.iloc[test_indices]

        # Compare actual class proportions with target proportions
        train_distribution = class_proportions(
            candidate_train, classes
        )
        test_distribution = class_proportions(
            candidate_test, classes
        )

        target = target_distribution.to_numpy()

        # Penalize differences in class balance
        class_error = (
            np.sum((train_distribution - target) ** 2)
            + np.sum((test_distribution - target) ** 2)
        )

        # Penalize differences from the desired split size
        size_error = (
            (len(candidate_test) / len(df) - test_size) ** 2
        )

        score = class_error + size_error

        if score < best_score:
            best_score = score
            best_indices = (train_indices, test_indices)

    return best_indices


def main():

    # 1. Load dataset
    df = pd.read_csv(INPUT_FILE)

    print("Original products:", len(df))

    # 2. Normalize ingredient descriptions
    groups = normalize_ingredients(df)

    # 3. Define the original NOVA class distribution
    target_distribution = (
        df["nova_group"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nOriginal NOVA distribution:")
    print(target_distribution)

    # 4. Split off the test set
    train_val_indices, test_indices = find_best_split(
        df,
        groups,
        test_size=0.15,
        target_distribution=target_distribution,
        seed=RANDOM_STATE
    )

    train_val_df = df.iloc[train_val_indices].copy()
    test_df = df.iloc[test_indices].copy()

    train_val_groups = groups.iloc[train_val_indices]

    # 5. Split training and validation
    train_indices, val_indices = find_best_split(
        train_val_df,
        train_val_groups,
        test_size=0.15 / 0.85,
        target_distribution=target_distribution,
        seed=RANDOM_STATE + 1
    )

    train_df = train_val_df.iloc[train_indices].copy()
    val_df = train_val_df.iloc[val_indices].copy()

    # 6. Save datasets
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    # 7. Check for ingredient-description leakage
    train_groups = set(groups.loc[train_df.index])
    val_groups = set(groups.loc[val_df.index])
    test_groups = set(groups.loc[test_df.index])

    print("\n========== SPLIT SUMMARY ==========")
    print("Training products:", len(train_df))
    print("Validation products:", len(val_df))
    print("Testing products:", len(test_df))

    print("\n========== NOVA DISTRIBUTION ==========")

    for name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df)
    ]:
        print(f"\n{name}:")
        print(
            split_df["nova_group"]
            .value_counts()
            .sort_index()
        )

        print("\nClass percentages:")
        print(
            split_df["nova_group"]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

    # 8. Verify no ingredient description crosses splits
    train_val_overlap = len(train_groups & val_groups)
    train_test_overlap = len(train_groups & test_groups)
    val_test_overlap = len(val_groups & test_groups)

    print("\n========== DATA LEAKAGE CHECK ==========")
    print("Training-validation overlap:", train_val_overlap)
    print("Training-testing overlap:", train_test_overlap)
    print("Validation-testing overlap:", val_test_overlap)

    if (
        train_val_overlap == 0
        and train_test_overlap == 0
        and val_test_overlap == 0
    ):
        print("\nSUCCESS: No ingredient-description leakage!")
    else:
        print("\nWARNING: Ingredient-description leakage detected!")

    print("\nDataset splitting complete!")


if __name__ == "__main__":
    main()
