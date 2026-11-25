
from pathlib import Path
import time
import statistics

import torch

from src.deep_learning.inference.predict import (
    load_model,
    predict_nova,
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

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "deep_learning"
    / "reports"
)

SAMPLE_TEXT = (
    "whole wheat flour, sugar, vegetable oil, "
    "salt, raising agents, natural flavouring"
)

WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


# --------------------------------------------------
# 2. Calculate model storage size
# --------------------------------------------------

def calculate_model_size():

    total_size = 0

    for file in MODEL_DIR.rglob("*"):

        if file.is_file():
            total_size += file.stat().st_size

    return total_size / (1024 * 1024)


# --------------------------------------------------
# 3. Benchmark model
# --------------------------------------------------

def main():

    print("=" * 55)
    print("DISTILBERT PERFORMANCE BENCHMARK")
    print("=" * 55)

    # Model size.
    model_size = calculate_model_size()

    print(f"\nSaved model directory size: {model_size:.2f} MB")

    # Loading time.
    print("\nLoading tokenizer and model...")

    start_time = time.perf_counter()

    tokenizer, model = load_model()

    loading_time = time.perf_counter() - start_time

    print(f"Model loading time: {loading_time:.2f} seconds")
    print(f"Execution device: {model.device}")

    # Warm-up runs.
    print("\nPerforming warm-up predictions...")

    for _ in range(WARMUP_RUNS):

        predict_nova(
            SAMPLE_TEXT,
            tokenizer,
            model,
        )

    if model.device.type == "mps":
        torch.mps.synchronize()

    # Benchmark runs.
    print(f"\nRunning {BENCHMARK_RUNS} benchmark predictions...")

    prediction_times = []

    for run in range(BENCHMARK_RUNS):

        if model.device.type == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()

        result = predict_nova(
            SAMPLE_TEXT,
            tokenizer,
            model,
        )

        if model.device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start_time

        prediction_times.append(elapsed)

        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{BENCHMARK_RUNS}")

    # Results.
    average_time = statistics.mean(prediction_times)
    median_time = statistics.median(prediction_times)
    minimum_time = min(prediction_times)
    maximum_time = max(prediction_times)

    print("\n" + "=" * 55)
    print("BENCHMARK RESULTS")
    print("=" * 55)

    print(f"Model size: {model_size:.2f} MB")
    print(f"Model loading time: {loading_time:.2f} seconds")
    print(f"Device: {model.device}")

    print(f"\nAverage inference time: {average_time * 1000:.2f} ms")
    print(f"Median inference time: {median_time * 1000:.2f} ms")
    print(f"Minimum inference time: {minimum_time * 1000:.2f} ms")
    print(f"Maximum inference time: {maximum_time * 1000:.2f} ms")

    print(
        f"\nApproximate throughput: "
        f"{1 / average_time:.2f} predictions/second"
    )

    print("\nSample prediction:")
    print("Predicted NOVA:", result["predicted_nova"])
    print("Confidence:", result["confidence"], "%")

    # Save report.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORT_DIR / "model_performance_benchmark.txt"

    with open(report_file, "w") as file:

        file.write("DISTILBERT PERFORMANCE BENCHMARK\n")
        file.write("=" * 40 + "\n")

        file.write(f"Model size: {model_size:.2f} MB\n")
        file.write(f"Loading time: {loading_time:.2f} seconds\n")
        file.write(f"Device: {model.device}\n")

        file.write(
            f"Average inference: {average_time * 1000:.2f} ms\n"
        )

        file.write(
            f"Median inference: {median_time * 1000:.2f} ms\n"
        )

        file.write(
            f"Minimum inference: {minimum_time * 1000:.2f} ms\n"
        )

        file.write(
            f"Maximum inference: {maximum_time * 1000:.2f} ms\n"
        )

        file.write(
            f"Throughput: {1 / average_time:.2f} predictions/second\n"
        )

    print("\nReport saved to:")
    print(report_file)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
