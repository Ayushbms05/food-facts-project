
from pathlib import Path

import pandas as pd

from src.deep_learning.inference.predict import load_model, predict_nova


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "datasets"
    / "test.csv"
)

SAMPLE_SIZE = 10


def main():

    print("Loading test dataset...")

    test_df = pd.read_csv(TEST_PATH)

    # Select 10 reproducible random test products.
    sample_df = test_df.sample(
        n=min(SAMPLE_SIZE, len(test_df)),
        random_state=42,
    )

    tokenizer, model = load_model()

    correct = 0

    print("\nSAVED MODEL PREDICTION TEST")
    print("=" * 55)

    for index, row in sample_df.iterrows():

        actual = int(row["nova_group"])

        result = predict_nova(
            str(row["ingredient_text"]),
            tokenizer,
            model,
        )

        predicted = result["predicted_nova"]

        is_correct = predicted == actual

        if is_correct:
            correct += 1

        print(f"\nProduct: {row['code']}")
        print(f"Actual NOVA: NOVA {actual}")
        print(f"Predicted: {result['label']}")
        print(f"Confidence: {result['confidence']}%")
        print("Result:", "CORRECT" if is_correct else "INCORRECT")

    accuracy = correct / len(sample_df) * 100

    print("\n" + "=" * 55)
    print(f"Correct predictions: {correct}/{len(sample_df)}")
    print(f"Sample accuracy: {accuracy:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
