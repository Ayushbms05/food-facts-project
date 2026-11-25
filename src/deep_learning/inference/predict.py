
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "models"
    / "distilbert_food_nova"
)

MAX_LENGTH = 256

LABEL_NAMES = [
    "NOVA 1",
    "NOVA 2",
    "NOVA 3",
    "NOVA 4",
]


# --------------------------------------------------
# 2. Select device
# --------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# --------------------------------------------------
# 3. Load saved model and tokenizer
# --------------------------------------------------

def load_model():

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_DIR}"
        )

    print(f"Loading model on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    if model.config.num_labels != 4:
        raise ValueError(
            "Expected a four-class NOVA classification model."
        )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully!")

    return tokenizer, model


# --------------------------------------------------
# 4. Predict NOVA group
# --------------------------------------------------

def predict_nova(ingredient_text, tokenizer, model):

    if not isinstance(ingredient_text, str):
        raise TypeError(
            "Ingredient text must be a string."
        )

    ingredient_text = ingredient_text.strip()

    if not ingredient_text:
        raise ValueError(
            "Ingredient text cannot be empty."
        )

    if model.config.num_labels != 4:
        raise ValueError(
            "Model must have exactly four NOVA output classes."
        )

    # Convert ingredient text into model inputs.
    inputs = tokenizer(
        ingredient_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Move inputs to the model's device.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # Generate prediction without calculating gradients.
    model.eval()

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=1,
        )[0]

    # Find the class with the highest probability.
    predicted_index = probabilities.argmax().item()

    # Convert class index 0-3 into NOVA group 1-4.
    predicted_nova = predicted_index + 1

    confidence = probabilities[predicted_index].item()

    class_probabilities = {
        LABEL_NAMES[i]: round(
            probabilities[i].item() * 100,
            2,
        )
        for i in range(4)
    }

    return {
        "predicted_nova": predicted_nova,
        "label": LABEL_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2),
        "class_probabilities": class_probabilities,
    }


# --------------------------------------------------
# 5. Run prediction from terminal
# --------------------------------------------------

if __name__ == "__main__":

    tokenizer, model = load_model()

    print("\nDistilBERT Food NOVA Predictor")
    print("Enter 'exit' to quit.\n")

    while True:

        ingredient_text = input(
            "Enter ingredient text: "
        )

        if ingredient_text.strip().lower() == "exit":
            print("Exiting predictor.")
            break

        try:

            result = predict_nova(
                ingredient_text,
                tokenizer,
                model,
            )

            print("\nPrediction:", result["label"])
            print("NOVA Group:", result["predicted_nova"])

            print(
                "Model confidence:",
                f'{result["confidence"]}%',
            )

            print("\nClass probabilities:")

            for label, probability in (
                result["class_probabilities"].items()
            ):
                print(f"{label}: {probability}%")

            print("-" * 40)

        except (ValueError, TypeError) as error:

            print("Input error:", error)
