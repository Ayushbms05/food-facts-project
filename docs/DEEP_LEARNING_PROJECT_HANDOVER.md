# FOOD HEALTH PREDICTION PROJECT

# DEEP LEARNING — FINAL PROJECT HANDOVER

**Last Updated:** 24 November 2026
**Workstream:** Member 2 — Deep Learning / Machine Learning
**Current Status:** Core model development, evaluation, inference interface and performance benchmarking completed. Final documentation, Git verification and application/backend integration remain.

---

# 1. PROJECT OVERVIEW

The Food Health Prediction project is designed to help users understand the processing category of food products using their ingredient information.

The project consists of three major workstreams:

1. **Big Data Analytics (BDA):** Processing and integrating large food-product datasets, including Open Food Facts (OFF) and USDA data, using Apache Spark.
2. **Deep Learning / Machine Learning:** Predicting a food product's NOVA processing group (1–4) using its ingredient text.
3. **Mobile Application:** Developing an Android-first application that can identify food products and display product information and NOVA predictions.

The primary research question for the Deep Learning workstream is:

> Can a model predict a product's NOVA group from its ingredient text?

## 1.1 Understanding NOVA classification

NOVA classifies foods according to the nature and extent of food processing.

It is not simply a healthy-versus-unhealthy classification.

- **NOVA 1:** Unprocessed or minimally processed foods.
- **NOVA 2:** Processed culinary ingredients.
- **NOVA 3:** Processed foods.
- **NOVA 4:** Ultra-processed foods.

The classification may depend on ingredients, manufacturing processes and product context.

## 1.2 Role of Deep Learning in the project

The DL component receives ingredient text and predicts one of four NOVA groups.

Its responsibility is to:

- Prepare ingredient-text datasets.
- Investigate dataset quality.
- Train and compare classification models.
- Select the best-performing model.
- Evaluate model performance.
- Provide a reusable prediction interface.
- Return NOVA predictions and class probabilities.

The DL component does not independently identify a product from its barcode or retrieve product information from external databases.

## 1.3 Important research limitation

Ingredient text alone may not contain all information needed to assign a correct NOVA group.

Two products with identical or similar ingredient descriptions may have different NOVA labels because manufacturing processes or additional product context are missing.

The model learns patterns from the available dataset labels. Its output should therefore be treated as a prediction, not a guaranteed or officially verified NOVA classification.

---

# 2. TEAM AND WORKSTREAM CONTEXT

The project has three main components.

| Workstream         | Responsibility                                           |
| ------------------ | -------------------------------------------------------- |
| Big Data Analytics | Dataset processing, cleaning and integration using Spark |
| Deep Learning      | Ingredient-text classification and model development     |
| Mobile Application | Android interface and product identification             |

The intended high-level workflow is:

1. A user scans a food product's barcode.
2. The application or backend identifies the product.
3. Product information and ingredient text are retrieved from the product database.
4. Ingredient text is passed to the DL prediction interface.
5. DistilBERT predicts a NOVA group.
6. The prediction is returned to the application.
7. The application displays the product information and predicted NOVA group.

**Important:** This describes the intended architecture. Complete end-to-end integration has not yet been verified.

## 2.1 Historical BDA results

The BDA workstream previously reported the following integration results:

| Metric                                | Reported value |
| ------------------------------------- | -------------: |
| Matched products                      |        368,055 |
| Products with OFF ingredients         |        314,306 |
| Products with USDA ingredients        |        365,741 |
| Products with both ingredient sources |        313,501 |

These are teammate-reported historical results and have not been independently re-verified by the DL workstream.

Previously mentioned BDA output paths:

```text
data/integration/combined_products.parquet
data/integration/usda_latest.parquet
```

The DL experiments documented here were conducted using:

```text
data/food.parquet
```

The BDA integrated files should not be treated as the DL training source unless their existence, schema and provenance have been verified.

---

# 3. REPOSITORY AND DEVELOPMENT ENVIRONMENT

## 3.1 Repository

Repository:

```text
food-facts-project/
```

DL working branch:

```text
feature/deep-learning
```

Shared team branch:

```text
main
```

DL development should be maintained on `feature/deep-learning`. Changes should not be pushed directly to `main` without team agreement.

## 3.2 Development machine

The experiments were conducted on:

- MacBook Air with Apple M1 chip.
- 8 GB unified memory.
- Python 3.12.5 ARM64.
- macOS environment.
- Existing Python virtual environment: `.venv`.

## 3.3 Software versions

The versions verified in the local development environment are:

| Library      | Version |
| ------------ | ------- |
| PyTorch      | 2.15.0  |
| Transformers | 5.24.0  |
| NumPy        | 2.5.4   |
| Pandas       | 3.1.1   |
| scikit-learn | 1.9.1   |
| PyArrow      | 26.0.1  |
| Matplotlib   | 3.11.2  |
| Safetensors  | 0.8.0   |

These versions are recorded in:

```text
requirements-dl.txt
```

Installation command:

```bash
pip install -r requirements-dl.txt
```

The exact dependency versions were verified in the local environment. Compatibility should still be checked when installing on another operating system or hardware configuration.

## 3.4 Apple MPS acceleration

The machine supports PyTorch's MPS backend.

MPS stands for Metal Performance Shaders. It allows supported PyTorch operations to use Apple's GPU through Metal.

The following checks were previously successful:

```python
import torch

print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

MPS was used for CNN, BiLSTM and DistilBERT experiments.

Because the machine has only 8 GB of unified memory shared between CPU and GPU, smaller batch sizes and memory-conscious processing were important.

---

# 4. CURRENT DEEP LEARNING PROJECT STRUCTURE

The DL source code was reorganized into separate folders according to functionality.

```text
src/
└── deep_learning/
    │
    ├── analysis/
    │   ├── __init__.py
    │   ├── analyze_full_dataset.py
    │   ├── analyze_text_quality.py
    │   ├── audit_ingredient_languages.py
    │   ├── check_dataset.py
    │   ├── investigate_duplicates.py
    │   ├── investigate_long_texts.py
    │   └── sample_long_texts.py
    │
    ├── data_preparation/
    │   ├── __init__.py
    │   ├── prepare_dataset.py
    │   ├── prepare_natural_sample.py
    │   └── split_dataset.py
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── analyze_natural_errors.py
    │   ├── analyze_prediction_confidence.py
    │   ├── analyze_model_calibration.py
    │   ├── benchmark_model_performance.py
    │   ├── evaluate_natural_distribution.py
    │   └── test_predictor.py
    │
    ├── inference/
    │   ├── __init__.py
    │   └── predict.py
    │
    └── models/
        ├── __init__.py
        ├── train_baseline.py
        ├── train_cnn.py
        ├── train_lstm.py
        └── train_transformer.py
```

The generated DL data is organized as follows:

```text
data/
└── deep_learning/
    │
    ├── datasets/
    │   ├── training_sample.csv
    │   ├── train.csv
    │   ├── validation.csv
    │   ├── test.csv
    │   └── natural_distribution_sample.csv
    │
    ├── models/
    │   ├── cnn_best.pt
    │   ├── cnn_vocab.json
    │   ├── lstm_best.pt
    │   ├── lstm_vocab.json
    │   └── distilbert_food_nova/
    │
    └── reports/
        ├── duplicate_analysis.csv
        ├── ingredient_language_audit.csv
        ├── long_text_review.csv
        ├── natural_distribution_predictions.csv
        ├── natural_distribution_errors.csv
        ├── calibration_reliability.csv
        ├── calibration_reliability.png
        ├── calibration_summary.txt
        └── model_performance_benchmark.txt
```

The precise presence of each generated file should be verified in the current checkout.

---

# 5. SOURCE DATASET

The original DL source dataset is:

```text
data/food.parquet
```

Previously scanned dataset size:

```text
4,895,629 rows
```

Important columns:

| Column        | Description                       |
| ------------- | --------------------------------- |
| `code`        | Product barcode/code              |
| `ingredients` | Structured ingredient information |
| `nova_group`  | NOVA classification label         |
| `lang`        | Product language metadata         |

The ingredient field is structured JSON-like information, rather than a simple text column.

It can contain:

- A `text` field.
- Nested ingredient objects.
- Nested ingredient lists.

The preparation script recursively extracts the text values.

## 5.1 Language distribution

Historical language counts from the full dataset scan:

| Language |  Products |
| -------- | --------: |
| English  | 2,054,773 |
| French   | 1,386,936 |
| Spanish  |   385,414 |
| German   |   360,625 |
| Italian  |   256,815 |
| Dutch    |   103,694 |

These figures relate to the previously examined dataset snapshot.

The `lang` field describes the product language and does not guarantee that every ingredient phrase is written in that language.

## 5.2 English-first development decision

The initial experiments used English-language products to establish a reproducible baseline.

This was a practical first step, not a decision to permanently restrict the project to English.

The longer-term direction remains:

1. Establish an English baseline.
2. Preserve original language information.
3. Expand multilingual evaluation.
4. Investigate Indian packaged foods and terminology.
5. Evaluate performance separately across languages and product populations.

Potential future models include mBERT and XLM-RoBERTa.

Neither has been trained or evaluated in the current experiments.

Potential Indian ingredient terms include:

- Atta
- Maida
- Besan
- Paneer
- Ghee
- Masalas

NOVA classification must not be inferred from a country's cuisine or an ingredient name alone.

---

# 6. DATASET PREPARATION

## 6.1 Preparation script

File:

```text
src/deep_learning/data_preparation/prepare_dataset.py
```

Purpose:

- Read the large Parquet dataset in batches.
- Avoid loading the entire dataset into Pandas.
- Filter English-language records.
- Remove records with invalid or missing NOVA labels.
- Extract ingredient text recursively.
- Sample up to 2,500 records from each NOVA class.
- Create a balanced 10,000-product dataset.

Implementation details:

| Setting                  | Value   |
| ------------------------ | ------- |
| Processing library       | PyArrow |
| Batch size               | 5,000   |
| `use_threads`            | False   |
| Random seed              | 42      |
| Maximum sample per class | 2,500   |

Output:

```text
data/deep_learning/datasets/training_sample.csv
```

Output columns:

```text
code, ingredient_text, nova_group
```

## 6.2 Ingredient extraction

The extraction function recursively processes structured ingredient data.

Its purpose is to collect ingredient names from both parent and nested ingredient objects.

This converts the structured source field into text suitable for classification.

## 6.3 Eligible English records

The full dataset scan reported:

| NOVA group | Eligible English products |
| ---------- | ------------------------: |
| NOVA 1     |                    53,594 |
| NOVA 2     |                     8,684 |
| NOVA 3     |                    91,175 |
| NOVA 4     |                   387,061 |
| **Total**  |               **540,514** |

The balanced sample contains 2,500 records per class, giving 10,000 records in total.

---

# 7. DATA QUALITY INVESTIGATION

Several investigations were performed before model training.

The purpose was to understand the data rather than blindly remove records.

## 7.1 Dataset validation

Script:

```text
src/deep_learning/analysis/check_dataset.py
```

Reported sample results:

| Check                                  |            Result |
| -------------------------------------- | ----------------: |
| Total rows                             |            10,000 |
| Columns                                |                 3 |
| Missing product codes                  |                 0 |
| Missing ingredient text                |                 0 |
| Missing NOVA labels                    |                 0 |
| Duplicate barcodes                     |                 0 |
| Initial repeated descriptions          |             2,315 |
| Unique descriptions                    |             7,685 |
| Descriptions shorter than 5 characters |               143 |
| Mean text length                       | 129.55 characters |
| Median text length                     |     56 characters |
| Maximum text length                    |  2,551 characters |

All four classes had 2,500 records in the balanced sample.

Short descriptions such as `egg`, `salt`, `oil` or `broccoli` can be valid ingredient descriptions.

They were therefore not automatically deleted.

## 7.2 Duplicate investigation

Script:

```text
src/deep_learning/analysis/investigate_duplicates.py
```

Normalization involved:

- Lowercasing.
- Removing leading and trailing whitespace.
- Collapsing repeated whitespace.

Reported findings:

```text
Total products: 10,000
Products with repeated descriptions: 3,382
Repeated unique descriptions: 551
Descriptions with conflicting NOVA labels: 73
```

Examples included:

- Agave nectar.
- Almonds.
- Avocado.
- Avocado oil.
- Balsamic vinegar.
- Arrowroot starch.

Some identical normalized ingredient descriptions were associated with different NOVA labels.

Possible explanations include missing manufacturing context, label inconsistency and data-quality issues.

These findings do not establish which label is correct.

**Decision:** Repeated descriptions were not automatically deleted. Instead, normalized ingredient descriptions were grouped during the train/validation/test splitting process to prevent identical descriptions crossing partitions.

## 7.3 Text-quality investigation

Script:

```text
src/deep_learning/analysis/analyze_text_quality.py
```

Reported results from the balanced sample:

| Measure                                 | Result |
| --------------------------------------- | -----: |
| Mean characters                         | 129.55 |
| Standard deviation                      | 196.81 |
| Minimum                                 |      3 |
| Median                                  |     56 |
| Maximum                                 |  2,551 |
| Mean words                              |  17.79 |
| Median words                            |      8 |
| Maximum words                           |    361 |
| Texts under 5 characters                |    143 |
| Texts over 1,000 characters             |     82 |
| Records containing numbers              |  1,699 |
| Records containing punctuation          |  7,297 |
| Records containing non-ASCII characters |    461 |
| Unusual whitespace                      |      0 |

Numbers, punctuation and non-ASCII characters were preserved because they may contain meaningful information.

## 7.4 Full-dataset analysis

Script:

```text
src/deep_learning/analysis/analyze_full_dataset.py
```

Reported results:

```text
Total rows: 4,895,629
Total English products: 2,054,773
Eligible English products: 540,514
Invalid or missing NOVA labels: 1,503,903
Missing ingredient text: 10,356
```

Natural NOVA distribution:

| NOVA group | Products | Percentage |
| ---------- | -------: | ---------: |
| NOVA 1     |   53,594 |      9.92% |
| NOVA 2     |    8,684 |      1.61% |
| NOVA 3     |   91,175 |     16.87% |
| NOVA 4     |  387,061 |     71.61% |

The dataset is strongly imbalanced.

NOVA 2 is particularly underrepresented, while NOVA 4 represents the majority of eligible English records.

## 7.5 Long-description investigation

Scripts:

```text
src/deep_learning/analysis/investigate_long_texts.py

src/deep_learning/analysis/sample_long_texts.py
```

Full-dataset text lengths:

| Measure                     |            Result |
| --------------------------- | ----------------: |
| Mean                        | 242.96 characters |
| Median                      |    179 characters |
| Minimum                     |       1 character |
| Maximum                     | 20,025 characters |
| 25th percentile             |                84 |
| 75th percentile             |               312 |
| Texts under 5 characters    |             2,877 |
| Texts over 1,000 characters |             9,330 |

Long-description class distribution:

| NOVA group | Long descriptions |
| ---------- | ----------------: |
| NOVA 1     |                 2 |
| NOVA 2     |                 0 |
| NOVA 3     |                19 |
| NOVA 4     |             9,309 |

Approximately 99.8% of descriptions longer than 1,000 characters belonged to NOVA 4.

A reproducible sample of 30 long descriptions was also inspected. All 30 selected descriptions belonged to NOVA 4.

Some contained:

- Multiple product varieties.
- Multilingual label content.
- Nutrition information.
- Storage and cooking instructions.
- Manufacturer information.
- OCR noise.
- Genuine complex ingredient lists.

**Decision:** Long descriptions were not automatically removed. Their influence remains an area for future investigation.

---

# 8. TRAIN, VALIDATION AND TEST SPLITTING

Script:

```text
src/deep_learning/data_preparation/split_dataset.py
```

The balanced 10,000-product sample was divided into three partitions.

The script used normalized ingredient descriptions as grouping keys and `GroupShuffleSplit` to avoid the same normalized ingredient description appearing in multiple partitions.

It searched 500 candidate splits and used fixed random seeds.

## 8.1 Split sizes

| Partition  |   Products |
| ---------- | ---------: |
| Training   |      7,018 |
| Validation |      1,481 |
| Test       |      1,501 |
| **Total**  | **10,000** |

## 8.2 Class distribution

| NOVA group | Training | Validation | Test |
| ---------- | -------: | ---------: | ---: |
| NOVA 1     |    1,742 |        375 |  383 |
| NOVA 2     |    1,758 |        369 |  373 |
| NOVA 3     |    1,767 |        368 |  365 |
| NOVA 4     |    1,751 |        369 |  380 |

## 8.3 Leakage verification

Reported verification:

```text
Train-validation overlap: 0
Train-test overlap: 0
Validation-test overlap: 0

SUCCESS: No ingredient-description leakage!
```

This establishes that normalized ingredient descriptions did not overlap across the three partitions in the reported split.

It does not establish that every product or barcode is independent across the partitions.

## 8.4 Important evaluation distinction

The balanced test set is useful for comparing models under equal class representation.

However, it does not represent the natural class distribution of the complete English dataset.

This is why a separate natural-distribution evaluation was subsequently performed.

---

# 9. MODEL DEVELOPMENT

Four classification experiments were completed.

All four models used the same balanced training, validation and test partitions.

The experiments were:

1. TF-IDF + Logistic Regression.
2. 1D CNN.
3. Bidirectional LSTM.
4. DistilBERT Transformer.

## 9.1 Experiment 1 — TF-IDF + Logistic Regression

Script:

```text
src/deep_learning/models/train_baseline.py
```

Category: Traditional Machine Learning.

TF-IDF configuration:

- Lowercase text.
- Unigrams and bigrams.
- Maximum features: 100,000.
- Minimum document frequency: 2.
- Sublinear term frequency enabled.

Logistic Regression configuration:

- Maximum iterations: 1,000.
- Random state: 42.

Results:

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     82.58% | 82.01% |
| Macro F1 |     0.8282 | 0.8188 |

Test classification results:

| Class  | Precision | Recall |   F1 |
| ------ | --------: | -----: | ---: |
| NOVA 1 |      0.86 |   0.94 | 0.90 |
| NOVA 2 |      0.92 |   0.62 | 0.74 |
| NOVA 3 |      0.64 |   0.84 | 0.72 |
| NOVA 4 |      0.95 |   0.88 | 0.91 |

Confusion matrix:

```text
[[360,   4,  14,   5],
 [ 16, 230, 124,   3],
 [ 36,  13, 305,  11],
 [  7,   2,  35, 336]]
```

This model established the traditional machine-learning baseline.

## 9.2 Experiment 2 — 1D CNN

Script:

```text
src/deep_learning/models/train_cnn.py
```

Framework: PyTorch.

Architecture:

1. Tokenization.
2. Vocabulary construction.
3. Embedding layer.
4. Three 1D convolution branches.
5. ReLU activation.
6. Global max pooling.
7. Dropout.
8. Fully connected four-class output layer.

Configuration:

| Setting                 |   Value |
| ----------------------- | ------: |
| Vocabulary size         |   3,695 |
| Maximum sequence length |     256 |
| Embedding dimension     |     128 |
| Convolution channels    |     128 |
| Kernel sizes            | 3, 5, 7 |
| Dropout                 |     0.4 |
| Batch size              |      64 |
| Learning rate           |   0.001 |
| Optimizer               |   AdamW |
| Maximum epochs          |      12 |
| Early stopping patience |       3 |

Results:

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     89.06% | 88.94% |
| Macro F1 |     0.8904 | 0.8889 |

Test classification results:

| Class  | Precision | Recall |   F1 |
| ------ | --------: | -----: | ---: |
| NOVA 1 |      0.89 |   0.87 | 0.88 |
| NOVA 2 |      0.87 |   0.91 | 0.89 |
| NOVA 3 |      0.86 |   0.84 | 0.85 |
| NOVA 4 |      0.94 |   0.93 | 0.94 |

Confusion matrix:

```text
[[335,  28,  14,   6],
 [ 18, 340,  12,   3],
 [ 22,  23, 305,  15],
 [  2,   1,  22, 355]]
```

Observations:

- Test accuracy was 88.94%.
- The CNN correctly classified 1,335 of 1,501 test products.
- NOVA 4 achieved an F1-score of 0.94.
- NOVA 3 remained comparatively challenging.
- Training accuracy reached 97.45%, suggesting some overfitting.

## 9.3 Experiment 3 — Bidirectional LSTM

Script:

```text
src/deep_learning/models/train_lstm.py
```

Framework: PyTorch.

Architecture:

1. Tokenization.
2. Vocabulary construction.
3. Embedding layer.
4. Forward LSTM.
5. Backward LSTM.
6. Combined bidirectional representation.
7. Dropout.
8. Fully connected four-class output layer.

The bidirectional architecture processes ingredient tokens in both directions to learn contextual relationships.

Configuration:

| Setting                   | Value |
| ------------------------- | ----: |
| Vocabulary size           | 3,695 |
| Maximum sequence length   |   256 |
| Embedding dimension       |   128 |
| Hidden size per direction |   128 |
| Dropout                   |   0.4 |
| Batch size                |    32 |
| Learning rate             | 0.001 |
| Optimizer                 | AdamW |
| Maximum epochs            |    12 |
| Early stopping patience   |     3 |

Results:

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     87.64% | 89.81% |
| Macro F1 |     0.8761 | 0.8977 |

Test classification results:

| Class  | Precision | Recall |   F1 |
| ------ | --------: | -----: | ---: |
| NOVA 1 |      0.89 |   0.93 | 0.91 |
| NOVA 2 |      0.94 |   0.89 | 0.91 |
| NOVA 3 |      0.83 |   0.84 | 0.84 |
| NOVA 4 |      0.94 |   0.92 | 0.93 |

Confusion matrix:

```text
[[358,   5,  16,   4],
 [ 19, 333,  20,   1],
 [ 23,  17, 307,  18],
 [  3,   0,  27, 350]]
```

Observations:

- Test accuracy was 89.81%.
- The model correctly classified 1,348 of 1,501 test products.
- NOVA 1 and NOVA 2 achieved F1-scores of 0.91.
- NOVA 3 remained the most challenging class.
- Training accuracy reached 97.89%, indicating a gap between training and validation performance.

## 9.4 Experiment 4 — DistilBERT Transformer

Script:

```text
src/deep_learning/models/train_transformer.py
```

Framework:

- PyTorch.
- Hugging Face Transformers.

Pretrained model:

```text
distilbert/distilbert-base-uncased
```

DistilBERT is a smaller pretrained Transformer derived from BERT.

It uses self-attention to learn contextual relationships between tokens.

Architecture:

1. DistilBERT tokenizer.
2. Token IDs and attention masks.
3. Pretrained DistilBERT encoder.
4. Classification head.
5. Four output classes corresponding to NOVA 1–4.

The pretrained model was fine-tuned for the four-class food-processing classification task.

Configuration:

| Setting                 |                   Value |
| ----------------------- | ----------------------: |
| Pretrained model        | DistilBERT base uncased |
| Maximum token length    |                     256 |
| Batch size              |                       8 |
| Gradient accumulation   |                       2 |
| Effective batch size    |                      16 |
| Learning rate           |                    2e-5 |
| Optimizer               |                   AdamW |
| Weight decay            |                    0.01 |
| Maximum epochs          |                       3 |
| Early stopping patience |                       2 |
| Random seed             |                      42 |

Training progress:

| Epoch | Training accuracy | Training macro F1 | Validation accuracy | Validation macro F1 |
| ----- | ----------------: | ----------------: | ------------------: | ------------------: |
| 1     |            80.01% |            0.8003 |              90.07% |              0.9003 |
| 2     |            90.71% |            0.9072 |              90.75% |              0.9071 |
| 3     |            93.10% |            0.9312 |              90.75% |              0.9078 |

The third epoch achieved the highest validation macro F1 and was selected for final testing.

Final results:

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     90.75% | 91.34% |
| Macro F1 |     0.9078 | 0.9132 |

Test classification results:

| Class  | Precision | Recall |   F1 |
| ------ | --------: | -----: | ---: |
| NOVA 1 |      0.92 |   0.94 | 0.93 |
| NOVA 2 |      0.87 |   0.93 | 0.90 |
| NOVA 3 |      0.87 |   0.85 | 0.86 |
| NOVA 4 |      0.99 |   0.92 | 0.96 |

Confusion matrix:

```text
[[360,  15,   8,   0],
 [  9, 348,  16,   0],
 [ 19,  32, 312,   2],
 [  2,   6,  21, 351]]
```

The model correctly classified 1,371 of 1,501 test products.

Observations:

- DistilBERT achieved the best overall test accuracy.
- It achieved the best overall macro F1.
- NOVA 4 achieved precision of 0.99.
- NOVA 1 achieved an F1-score of 0.93.
- NOVA 2 achieved an F1-score of 0.90.
- NOVA 3 remained the most challenging class, with an F1-score of 0.86.

---

# 10. FINAL MODEL COMPARISON

All four experiments used the same balanced test partition.

| Model                        | Test accuracy | Test macro F1 |
| ---------------------------- | ------------: | ------------: |
| TF-IDF + Logistic Regression |        82.01% |        0.8188 |
| 1D CNN                       |        88.94% |        0.8889 |
| Bidirectional LSTM           |        89.81% |        0.8977 |
| **DistilBERT**               |    **91.34%** |    **0.9132** |

DistilBERT improved test accuracy by:

- 9.33 percentage points over Logistic Regression.
- 2.40 percentage points over CNN.
- 1.53 percentage points over BiLSTM.

## 10.1 Model selection

**DistilBERT is the leading model based on the current balanced test-set results.**

It was selected as the current inference model because it achieved the highest test accuracy and macro F1 among the four experiments.

This does not establish that it is the best model for every real-world population or mobile deployment environment.

Model size, inference speed, class imbalance and deployment requirements must also be considered.

---

# 11. NATURAL-DISTRIBUTION EVALUATION

After the balanced model comparison, a separate evaluation was conducted using a sample that approximately represents the natural class distribution.

This was important because the full eligible English dataset is highly imbalanced.

## 11.1 Natural sample preparation

Script:

```text
src/deep_learning/data_preparation/prepare_natural_sample.py
```

Output:

```text
data/deep_learning/datasets/natural_distribution_sample.csv
```

The script:

- Scanned the original Parquet dataset.
- Filtered English records.
- Retained valid NOVA labels and ingredient text.
- Excluded normalized ingredient descriptions already present in the existing balanced splits.
- Used reservoir sampling.
- Selected 10,000 records with random seed 42.

Reported results:

| Measure                           |    Result |
| --------------------------------- | --------: |
| Source rows                       | 4,895,629 |
| Eligible records after exclusions |   485,831 |
| Excluded products                 |    54,683 |
| Natural sample size               |    10,000 |

The remaining eligible population had this distribution:

| NOVA group | Products | Percentage |
| ---------- | -------: | ---------: |
| NOVA 1     |   25,595 |      5.27% |
| NOVA 2     |    1,750 |      0.36% |
| NOVA 3     |   78,299 |     16.12% |
| NOVA 4     |  380,187 |     78.25% |

The selected sample contained:

| NOVA group |   Products | Percentage |
| ---------- | ---------: | ---------: |
| NOVA 1     |        562 |      5.62% |
| NOVA 2     |         35 |      0.35% |
| NOVA 3     |      1,615 |     16.15% |
| NOVA 4     |      7,788 |     77.88% |
| **Total**  | **10,000** |   **100%** |

**Important limitation:** The natural sample excluded previously seen normalized ingredient descriptions, but did not guarantee that all product barcodes were unseen. It is a same-source distribution audit, not a fully independent external validation dataset.

## 11.2 Natural-distribution evaluation

Script:

```text
src/deep_learning/evaluation/evaluate_natural_distribution.py
```

Results:

| Metric                | Result |
| --------------------- | -----: |
| Products evaluated    | 10,000 |
| Correct predictions   |  9,011 |
| Incorrect predictions |    989 |
| Accuracy              | 90.11% |
| Macro F1              | 0.6933 |

Class-wise results:

| Class  | Precision | Recall |   F1 | Support |
| ------ | --------: | -----: | ---: | ------: |
| NOVA 1 |      0.77 |   0.86 | 0.82 |     562 |
| NOVA 2 |      0.13 |   0.83 | 0.22 |      35 |
| NOVA 3 |      0.70 |   0.90 | 0.79 |   1,615 |
| NOVA 4 |      1.00 |   0.90 | 0.95 |   7,788 |

Confusion matrix:

```text
[[486,  38,  36,   2],
 [  2,  29,   4,   0],
 [ 59,  82, 1453, 21],
 [ 82,  77, 586, 7043]]
```

## 11.3 Interpretation

The natural-distribution accuracy is 90.11%, but macro F1 falls to 0.6933.

This difference is important.

NOVA 4 represents nearly 78% of the sample. Strong NOVA 4 performance contributes substantially to overall accuracy.

NOVA 2 represents only 35 of the 10,000 records. Its precision is particularly poor.

Therefore, accuracy alone is not sufficient to describe the model's real-world performance.

---

# 12. ERROR INVESTIGATION

Script:

```text
src/deep_learning/evaluation/analyze_natural_errors.py
```

Output:

```text
data/deep_learning/reports/natural_distribution_errors.csv
```

The investigation focused on two major error patterns.

## 12.1 False NOVA 2 predictions

There were 197 incorrect NOVA 2 predictions.

| Actual class | Incorrectly predicted as NOVA 2 |
| ------------ | ------------------------------: |
| NOVA 1       |                              38 |
| NOVA 3       |                              82 |
| NOVA 4       |                              77 |
| **Total**    |                         **197** |

Examples included descriptions involving:

- Peanuts and sea salt.
- Almonds and maltodextrin.
- Blue corn, water, lime and salt.
- Canola oil and additives.
- Dextrose and sweeteners.

The model appears to associate some ingredient patterns with NOVA 2 even when the supplied dataset label belongs to another class.

This requires further investigation and should not be corrected by simply changing predictions.

## 12.2 NOVA 4 predicted as NOVA 3

There were 586 cases where actual NOVA 4 products were predicted as NOVA 3.

Examples included:

- Complex Indian meal descriptions.
- Soy sauce and additives.
- Chips.
- Spice blends.
- Biscuits.
- Jam.
- Mints.
- Processed meat products.

Potential contributing factors include ingredient overlap, missing processing context, complex descriptions and label quality.

## 12.3 Duplicate findings in the natural sample

Reported results:

```text
Total products: 10,000
Products with repeated descriptions: 232
Repeated unique descriptions: 110
Repeated descriptions with conflicting labels: 1
```

Duplicate descriptions were therefore not the main explanation for the observed natural-sample errors.

The examples also indicated that product-language metadata does not always match the language of every ingredient phrase.

No automatic correction of these records has been implemented.

---

# 13. PREDICTION CONFIDENCE ANALYSIS

Script:

```text
src/deep_learning/evaluation/analyze_prediction_confidence.py
```

The prediction report was extended to store all four class probabilities and the highest class probability.

Output:

```text
data/deep_learning/reports/natural_distribution_predictions.csv
```

Additional columns:

```text
nova1_probability
nova2_probability
nova3_probability
nova4_probability
predicted_nova
confidence
correct
```

All class probabilities and confidence values are represented as percentages.

## 13.1 Confidence results

| Measure            | Correct predictions | Incorrect predictions |
| ------------------ | ------------------: | --------------------: |
| Count              |               9,011 |                   989 |
| Mean confidence    |              96.82% |                82.77% |
| Median confidence  |              99.23% |                89.51% |
| Minimum confidence |              36.76% |                32.64% |
| Maximum confidence |              99.71% |                99.54% |

The model generally assigns lower confidence to incorrect predictions.

However, some incorrect predictions still have very high confidence.

## 13.2 NOVA 2 confidence

False NOVA 2 predictions:

| Actual class | Count | Mean confidence |
| ------------ | ----: | --------------: |
| NOVA 1       |    38 |          74.79% |
| NOVA 3       |    82 |          80.77% |
| NOVA 4       |    77 |          77.35% |

Only 29 of the 226 total NOVA 2 predictions were correct.

This is approximately 12.8% precision for the predicted NOVA 2 class.

## 13.3 NOVA 4 predicted as NOVA 3

There were 586 such errors.

| Measure            | Result |
| ------------------ | -----: |
| Average confidence | 85.33% |
| Median confidence  | 93.10% |
| Minimum confidence | 34.15% |
| Maximum confidence | 98.95% |

This indicates that the model can be highly confident even when its predicted class is incorrect.

## 13.4 Confidence by predicted class

| Predicted class | Count | Mean confidence | Median confidence |
| --------------- | ----: | --------------: | ----------------: |
| NOVA 1          |   629 |          91.34% |            98.06% |
| NOVA 2          |   226 |          80.12% |            86.42% |
| NOVA 3          | 2,079 |          92.01% |            97.61% |
| NOVA 4          | 7,066 |          97.29% |            99.41% |

**Important:** Softmax confidence is a model probability score. It is not a guarantee that a prediction has that exact probability of being correct.

---

# 14. MODEL CALIBRATION ANALYSIS

Script:

```text
src/deep_learning/evaluation/analyze_model_calibration.py
```

The calibration analysis was performed using the 1,481-product validation partition.

The script calculated:

- Expected Calibration Error (ECE).
- Multiclass Brier score.
- Reliability table.
- Reliability diagram.

Generated files:

```text
data/deep_learning/reports/

├── calibration_reliability.csv
├── calibration_reliability.png
└── calibration_summary.txt
```

## 14.1 Calibration results

| Metric                     | Result |
| -------------------------- | -----: |
| Validation products        |  1,481 |
| Accuracy                   | 90.75% |
| Expected Calibration Error | 0.0427 |
| Multiclass Brier Score     | 0.1539 |

Lower ECE and Brier scores generally indicate better calibration.

## 14.2 Reliability findings

The 90–100% confidence bin contained 1,249 products.

| Measure            |                 Result |
| ------------------ | ---------------------: |
| Average confidence |                 98.13% |
| Actual accuracy    |                 94.32% |
| Difference         | 3.81 percentage points |

The model is somewhat overconfident in this high-confidence range.

The lower-confidence bins also show some variation between predicted confidence and observed accuracy.

## 14.3 Calibration limitation

The validation dataset was used during model selection.

Therefore, these calibration results are diagnostic rather than a final independent calibration assessment.

A separate untouched dataset should be used before making stronger claims about calibration.

Temperature scaling or other probability-calibration methods have not been implemented.

No calibrated confidence output is currently provided by the prediction interface.

---

# 15. FINAL INFERENCE / PREDICTION INTERFACE

A reusable prediction interface has now been implemented.

File:

```text
src/deep_learning/inference/predict.py
```

Saved model directory:

```text
data/deep_learning/models/distilbert_food_nova/
```

The model directory contains the trained model weights, configuration and matching tokenizer files.

The inference module uses:

- `AutoTokenizer`
- `AutoModelForSequenceClassification`
- PyTorch
- Softmax probability calculation

## 15.1 Device selection

The inference module selects the available device in this order:

1. Apple MPS.
2. CUDA.
3. CPU.

The model is moved to the selected device when loaded.

Input tensors are moved to the model's device before prediction.

## 15.2 Model loading

Function:

```python
load_model()
```

This function:

- Checks that the saved model directory exists.
- Loads the tokenizer.
- Loads the trained sequence-classification model.
- Verifies that the model has four output classes.
- Moves the model to the selected device.
- Sets the model to evaluation mode.

It returns:

```python
tokenizer, model
```

The model should be loaded once and reused for multiple predictions.

## 15.3 Prediction function

Function:

```python
predict_nova(ingredient_text, tokenizer, model)
```

Input:

```python
ingredient_text = (
    "whole wheat flour, sugar, vegetable oil, salt"
)
```

The function:

1. Validates that the input is a string.
2. Rejects empty ingredient text.
3. Tokenizes the ingredient description.
4. Truncates or pads the input to a maximum of 256 tokens.
5. Moves the input tensors to the model device.
6. Runs the model without gradient calculation.
7. Applies Softmax to the four output logits.
8. Selects the class with the highest probability.
9. Converts class index 0–3 into NOVA group 1–4.
10. Returns the prediction and class probabilities.

## 15.4 Output structure

The function returns a Python dictionary:

```python
{
    "predicted_nova": 4,
    "label": "NOVA 4",
    "confidence": 92.5,
    "class_probabilities": {
        "NOVA 1": 1.0,
        "NOVA 2": 1.5,
        "NOVA 3": 5.0,
        "NOVA 4": 92.5
    }
}
```

The numbers above are illustrative and not an actual model prediction.

The output fields are:

| Field                 | Meaning                                      |
| --------------------- | -------------------------------------------- |
| `predicted_nova`      | Predicted integer class from 1–4             |
| `label`               | Human-readable NOVA label                    |
| `confidence`          | Highest Softmax probability as a percentage  |
| `class_probabilities` | Probability percentages for all four classes |

## 15.5 Example usage

```python
from src.deep_learning.inference.predict import (
    load_model,
    predict_nova,
)

tokenizer, model = load_model()

ingredient_text = (
    "whole wheat flour, sugar, vegetable oil, salt"
)

result = predict_nova(
    ingredient_text,
    tokenizer,
    model,
)

print(result)
```

The tokenizer and model are loaded once and can be reused for multiple predictions.

## 15.6 Input validation

The function checks for:

- Non-string input.
- Empty or whitespace-only input.
- Models that do not have exactly four output classes.

These checks prevent common input and configuration errors.

## 15.7 Inference verification

The following command was successfully executed:

```bash
python -m src.deep_learning.inference.predict
```

The interactive prediction interface loaded the model and accepted ingredient text.

The saved-model test was also executed:

```bash
python -m src.deep_learning.evaluation.test_predictor
```

Reported result:

```text
Correct predictions: 10/10
Sample accuracy: 100.00%
```

This is a 10-product smoke test demonstrating that the saved model and inference interface work together.

It does not replace the full held-out test accuracy of 91.34%.

---

# 16. MODEL SIZE AND INFERENCE PERFORMANCE

Script:

```text
src/deep_learning/evaluation/benchmark_model_performance.py
```

The benchmark was conducted on the local Apple M1 MacBook Air using PyTorch MPS.

It measured:

- Saved model directory size.
- Model loading time.
- Prediction latency.
- Average inference time.
- Median inference time.
- Minimum and maximum inference time.
- Approximate throughput.

The benchmark included five warm-up predictions followed by 50 measured predictions.

## 16.1 Benchmark results

| Metric                 |                   Result |
| ---------------------- | -----------------------: |
| Model directory size   |                256.11 MB |
| Model loading time     |             2.48 seconds |
| Execution device       |                      MPS |
| Average inference time |                 57.26 ms |
| Median inference time  |                 51.20 ms |
| Minimum inference time |                 32.75 ms |
| Maximum inference time |                205.83 ms |
| Approximate throughput | 17.46 predictions/second |

Report:

```text
data/deep_learning/reports/model_performance_benchmark.txt
```

## 16.2 Interpretation

The model produced predictions in an average of approximately 76 milliseconds on the tested M1 machine.

The saved model directory occupies approximately 256 MB.

These measurements show that the model is usable for local inference on the tested development machine.

They do not establish performance on an Android phone.

## 16.3 Deployment recommendation

The initial deployment recommendation is to use backend inference.

In this arrangement:

1. Android sends ingredient text to a backend API.
2. The backend loads the DistilBERT model.
3. The backend performs prediction.
4. The backend returns the NOVA group and probability scores.
5. Android displays the result.

Reasons for this provisional recommendation:

- The existing model is approximately 256 MB.
- The current inference implementation uses PyTorch and Hugging Face Transformers.
- Backend model updates are easier to manage.
- Mobile conversion and compatibility have not been tested.

Direct Android inference remains a possible future option, but requires model conversion, a compatible mobile runtime and actual Android-device benchmarking.

No backend API has been implemented or verified yet.

---

# 17. CURRENT ACHIEVEMENTS

The following work has been completed in the DL workstream.

## Dataset and preparation

- Inspected the original Parquet dataset.
- Implemented memory-conscious PyArrow batch processing.
- Extracted nested ingredient text.
- Created a balanced 10,000-product English sample.
- Investigated missing values and duplicate barcodes.
- Investigated repeated ingredient descriptions.
- Identified descriptions with conflicting NOVA labels.
- Created leakage-aware train, validation and test partitions.
- Verified zero normalized ingredient-description overlap between partitions.
- Analyzed ingredient-text quality.
- Scanned the complete eligible English dataset.
- Investigated long ingredient descriptions.
- Created a reproducible long-description review sample.

## Model development

- Implemented TF-IDF + Logistic Regression.
- Trained a 1D CNN.
- Trained a Bidirectional LSTM.
- Fine-tuned DistilBERT.
- Compared all four models using the same balanced test partition.
- Selected DistilBERT as the leading current model candidate.

## Model evaluation

- Evaluated the selected model on the balanced test set.
- Prepared a natural-distribution sample.
- Evaluated the model on 10,000 naturally distributed records.
- Investigated natural-distribution errors.
- Generated confidence and class-probability reports.
- Performed initial calibration analysis.
- Generated a reliability table and diagram.

## Inference and performance

- Implemented a reusable inference interface.
- Added input validation.
- Verified four-class model configuration.
- Loaded the saved tokenizer and model.
- Tested prediction output.
- Successfully completed a 10-product inference smoke test.
- Measured model size and inference latency.

---

# 18. KNOWN LIMITATIONS AND UNFINISHED WORK

The project has achieved a functional DL model, but several limitations remain.

## 18.1 Dataset limitations

- The training dataset was balanced artificially.
- The full English dataset is strongly imbalanced.
- NOVA 2 represents only 1.61% of eligible English records.
- Some ingredient descriptions have conflicting NOVA labels.
- Ingredient text may omit important processing information.
- The dataset is based on a single source snapshot.

## 18.2 Model limitations

- The natural-distribution macro F1 is 0.6933.
- NOVA 2 precision is only 0.13 in the natural-distribution evaluation.
- NOVA 4 is frequently confused with NOVA 3.
- Some incorrect predictions have very high Softmax confidence.
- Long descriptions may contain dataset-specific signals.
- The model has not undergone independent external validation.

## 18.3 Language limitations

The current experiments are English-first.

The following have not been demonstrated:

- Hindi model performance.
- Regional Indian language performance.
- Multilingual model performance.
- India-specific classification performance.

## 18.4 Deployment limitations

The following have not been completed:

- Backend API implementation.
- Android integration.
- Direct mobile model conversion.
- Android-device performance testing.
- Production hosting.
- End-to-end barcode-to-prediction testing.

## 18.5 Calibration limitations

The model produces raw Softmax scores.

Formal probability calibration has not been applied.

The current validation ECE and Brier score are diagnostic results, not independent final calibration claims.

---

# 19. FUTURE IMPROVEMENTS

The following improvements are recommended after the core project handover.

## 19.1 Improve NOVA 2 performance

- Investigate false NOVA 2 predictions.
- Obtain more reliable labelled NOVA 2 examples.
- Evaluate class-weighted training.
- Investigate resampling strategies.
- Consider targeted data-quality review.

Any improvement should be evaluated on an untouched test dataset.

## 19.2 Improve natural-distribution performance

- Use representative naturally distributed training data.
- Investigate class imbalance.
- Evaluate per-class precision and recall.
- Study the effect of long descriptions.
- Evaluate generalization on new product records.

## 19.3 Multilingual expansion

Potential future work:

- mBERT.
- XLM-RoBERTa.
- Language-specific evaluation.
- Hindi and English evaluation.
- Additional regional languages when suitable labelled data is available.

## 19.4 India-focused evaluation

Future work should include:

- Indian packaged food products.
- Indian ingredient terminology.
- Suitable labelled datasets.
- Separate performance reporting for Indian products.
- Evaluation across Hindi and English first.

## 19.5 Probability calibration

Potential future work:

- Temperature scaling.
- Independent calibration data.
- Reliability diagrams on an untouched dataset.
- Evaluation of calibrated confidence outputs.

## 19.6 Deployment

- Implement backend inference.
- Define an API input/output contract.
- Connect the BDA product lookup to the inference interface.
- Connect the backend to Android.
- Evaluate on-device alternatives if needed.

---

# 20. CURRENT COMPLETION STATUS

| Component                        | Status                       |
| -------------------------------- | ---------------------------- |
| Source dataset inspection        | Completed                    |
| Ingredient extraction            | Completed                    |
| Balanced dataset preparation     | Completed                    |
| Data quality investigation       | Completed                    |
| Duplicate investigation          | Completed                    |
| Train/validation/test splitting  | Completed                    |
| Text-quality analysis            | Completed                    |
| Full-dataset analysis            | Completed                    |
| Long-text investigation          | Completed                    |
| Logistic Regression baseline     | Completed                    |
| 1D CNN                           | Completed                    |
| Bidirectional LSTM               | Completed                    |
| DistilBERT                       | Completed                    |
| Model comparison                 | Completed                    |
| Natural-distribution evaluation  | Completed                    |
| Error investigation              | Completed                    |
| Confidence analysis              | Completed                    |
| Calibration analysis             | Completed                    |
| Prediction interface             | Implemented and smoke-tested |
| Model performance benchmark      | Completed                    |
| Dependency file                  | Created and versions pinned  |
| Final documentation              | Updating                     |
| Git commit and push verification | Pending                      |
| Backend API                      | Not implemented              |
| BDA/DL application integration   | Not verified                 |
| Android integration              | Not implemented              |
| Multilingual evaluation          | Pending                      |
| India-specific evaluation        | Pending                      |

---

# 21. GIT AND VERSION CONTROL

The DL workstream uses:

```text
feature/deep-learning
```

All intended DL changes should be committed and pushed to this branch.

The latest previously shared VS Code screenshot showed approximately 44 Source Control changes.

The current Git status and remote commit history have not yet been verified after the latest changes.

Therefore, do not assume all scripts and documents have been committed or pushed.

Before committing, inspect:

```bash
git status
```

And:

```bash
git log -5 --oneline
```

Review all modified, deleted and untracked files.

Files generally excluded from Git:

- `.venv/`
- `__pycache__/`
- Raw datasets.
- Generated Parquet files.
- Large generated CSV datasets.
- Model checkpoints and caches unless an explicit sharing strategy is agreed.
- Temporary organization scripts.

The DistilBERT model directory is approximately 256 MB. A deliberate artifact-sharing strategy is required so teammates can obtain the trained model.

Only commit files that are intended to be part of the project.

Example commit workflow:

```bash
git status

git add requirements-dl.txt
git add src/deep_learning/
git add docs/

git status

git commit -m "Complete deep learning model development and evaluation"

git push origin feature/deep-learning

git status
```

The exact staging list must be adjusted after inspecting the actual Git changes.

Do not push directly to `main`.

---

# 22. IMMEDIATE NEXT STEPS

The recommended remaining work is:

1. Finalize this handover and the model experiment results document.
2. Review the repository's Git status.
3. Remove unnecessary temporary helper scripts after verifying they are no longer required.
4. Verify `.gitignore`.
5. Decide how the trained DistilBERT model will be shared with teammates.
6. Commit and push the intended DL code and documentation.
7. Give BDA the model, inference interface and dependency instructions.
8. Agree on the input/output contract for integration.
9. Implement backend inference when the application architecture is ready.
10. Integrate and test the Android application.
11. Continue multilingual and India-focused improvements later.

---

# 23. GUIDANCE FOR FUTURE DEVELOPMENT

The DL workstream should be continued with the following principles:

1. Preserve the original source dataset.
2. Do not automatically delete short, long, repeated or conflicting ingredient descriptions.
3. Verify the data source and schema before changing dataset inputs.
4. Use the same documented splits when reproducing the existing experiments.
5. Clearly distinguish balanced-test results from natural-distribution results.
6. Report class-wise metrics alongside overall accuracy.
7. Do not present Softmax confidence as a guaranteed probability of correctness.
8. Do not claim multilingual or India-specific performance without testing.
9. Do not claim Android or backend integration until it has been implemented and verified.
10. Avoid unnecessary retraining when the existing checkpoint and inference interface are working.
11. Keep generated datasets and model artifacts out of Git unless the team agrees otherwise.
12. Preserve the existing model results and their limitations when preparing presentations or reports.

The user prefers clear, step-by-step technical guidance with explicit VS Code and terminal instructions. Explain changes before asking them to execute commands.

---

# 24. FINAL CONCLUSION

The Deep Learning workstream has progressed from initial dataset inspection to a trained and evaluated four-class food-processing classification model.

Four approaches were implemented and compared:

- TF-IDF + Logistic Regression.
- 1D CNN.
- Bidirectional LSTM.
- DistilBERT Transformer.

DistilBERT achieved the strongest performance on the balanced held-out test set, with:

- 91.34% test accuracy.
- 0.9132 test macro F1.

A separate natural-distribution evaluation achieved:

- 90.11% accuracy.
- 0.6933 macro F1.

This revealed the importance of class imbalance, particularly the weak precision for NOVA 2.

Further work included:

- Duplicate and label-quality investigation.
- Long-description analysis.
- Prediction error investigation.
- Confidence analysis.
- Calibration diagnostics.
- Inference interface development.
- Model size and latency benchmarking.

The selected model has a saved checkpoint and tokenizer, and a reusable prediction function is available.

The model's measured directory size was 256.11 MB, its loading time was 2.48 seconds, and its average single-text inference time was 57.26 milliseconds on the tested Apple M1 machine using MPS. The median inference time was 51.20 milliseconds, with an approximate throughput of 17.46 predictions per second.

The core DL model development and evaluation are complete. The remaining focus is documentation, version-control verification, model sharing, backend integration and eventual Android application integration.

The current results provide a useful research and development foundation, but they do not establish perfect NOVA classification, independent external generalization, multilingual performance or production readiness.

**END OF DEEP LEARNING HANDOVER**
