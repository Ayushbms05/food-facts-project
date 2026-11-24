# Food Health Prediction Project — Deep Learning Handover

**Last updated:** 24 November 2026  
**Workstream:** Member 2 — Deep Learning / Machine Learning  
**Current status:** Four model experiments completed; model comparison completed; deployment and application integration remain.

This document provides a comprehensive handover of the Deep Learning (DL) workstream. It records the project's purpose, dataset investigations, preprocessing, model architectures, experiment results, decisions, limitations, and remaining tasks.

**Status discipline:** Distinguish completed work from planned work. Do not assume files have been committed or pushed without checking Git. Do not claim Android integration or real-world model performance has been verified when it has not.

---

# 1. Project Overview

The Food Health Prediction project has three main workstreams:

1. **Big Data Analytics (BDA):** Process and integrate large food-product datasets, including Open Food Facts (OFF) and USDA data, using Apache Spark.
2. **Deep Learning / Machine Learning:** Predict a food product's NOVA processing group (1–4) using ingredient text.
3. **Mobile Application:** Develop an Android-first application that can identify food products and display relevant product information and NOVA predictions.

The current DL research question is:

> Can a model predict a product's NOVA group from its ingredient text?

## What NOVA means

NOVA classifies foods according to the nature and extent of food processing. It is not simply a healthy-versus-unhealthy classification.

- **NOVA 1:** Unprocessed or minimally processed foods.
- **NOVA 2:** Processed culinary ingredients.
- **NOVA 3:** Processed foods.
- **NOVA 4:** Ultra-processed foods.

The exact classification can depend on ingredients, manufacturing processes, and product context.

## Important project limitation

Ingredient text alone may not contain all the information required to assign a correct NOVA group. Two products with identical or similar ingredient descriptions may have different labels because manufacturing details or other context are missing.

The model learns patterns associated with the available dataset labels. Its predictions should not be presented as guaranteed NOVA classifications.

---

# 2. Team and Workstream Context

- **BDA:** Apache Spark processing and integration of OFF and USDA data.
- **DL:** Ingredient-text extraction, dataset preparation, quality investigation, model training and evaluation.
- **Mobile application:** Android-first application development.

## Historical BDA integration results

The following figures were reported by the BDA workstream in shared project conversations. They have not been independently re-verified in this DL workstream.

| Metric                                | Reported value |
| ------------------------------------- | -------------: |
| Matched products                      |        368,055 |
| Products with OFF ingredients         |        314,306 |
| Products with USDA ingredients        |        365,741 |
| Products with both ingredient sources |        313,501 |

Historical output paths mentioned:

```text
data/integration/combined_products.parquet
data/integration/usda_latest.parquet
```

Do not assume these files exist in the current DL checkout. Verify their existence, schema, provenance and branch before changing the DL dataset source.

The DL experiments documented below were performed using `data/food.parquet`, not a verified BDA integrated output.

---

# 3. Repository, Branch and Environment

## Repository

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

DL changes should be developed and pushed to `feature/deep-learning`, not directly to `main`.

## Observed project structure

```text
food-facts-project/
├── src/
│   ├── bda/
│   └── deep_learning/
│       ├── prepare_dataset.py
│       ├── check_dataset.py
│       ├── investigate_duplicates.py
│       ├── split_dataset.py
│       ├── analyze_text_quality.py
│       ├── analyze_full_dataset.py
│       ├── investigate_long_texts.py
│       ├── sample_long_texts.py
│       ├── train_baseline.py
│       ├── train_cnn.py
│       ├── train_lstm.py
│       └── train_transformer.py
├── docs/
│   ├── DEEP_LEARNING_PROJECT_HANDOVER.md
│   └── MODEL_EXPERIMENT_RESULTS.md
├── data/
├── notebooks/
├── notes/
├── .venv/
├── .gitignore
└── README.md
```

The listed training and analysis scripts were present in the user's local checkout when the experiments were conducted. Their current Git tracking and remote status must be checked separately.

## Local machine

- MacBook Air with Apple M1 chip.
- 8 GB unified memory.
- Python 3.12.5 ARM64 was previously reported.
- Existing virtual environment: `.venv`.
- macOS 26.3 was previously reported.

Continue using the existing `.venv`; do not recreate it without a reason.

## Reported software versions

| Library / capability | Version or status            |
| -------------------- | ---------------------------- |
| PyTorch              | 2.15.0                       |
| Transformers         | 5.24.0                       |
| scikit-learn         | 1.9.1 (previously reported)  |
| Pandas               | 3.1.1 (previously reported)  |
| PyArrow              | 26.0.1 (previously reported) |
| PyTorch MPS          | Available                    |
| MPS build            | Available                    |

The following checks were successfully executed:

```python
import torch

print(torch.__version__)
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

Reported output:

```text
PyTorch: 2.15.0
MPS available: True
MPS built: True
```

An LSTM forward-pass test on MPS also succeeded.

## Apple MPS explanation

Apple Metal is Apple's GPU-computing technology. PyTorch provides Apple GPU support through the MPS (Metal Performance Shaders) backend.

This allowed the CNN, LSTM and DistilBERT experiments to use the M1 GPU.

The machine has only 8 GB of unified memory shared between CPU and GPU. Smaller batches and compact models are therefore important.

---

# 4. Source Dataset

Current DL source:

```text
data/food.parquet
```

Previously reported row count:

```text
4,895,629
```

Important columns:

| Column        | Description                                 |
| ------------- | ------------------------------------------- |
| `code`        | Product barcode/code                        |
| `ingredients` | Structured JSON-like ingredient information |
| `nova_group`  | NOVA label, expected values 1–4             |
| `lang`        | Product language code                       |

The `ingredients` field is structured data, not plain text. It can contain a `text` field and nested ingredient lists.

The preparation script recursively extracts ingredient text, including nested ingredients.

**Language caveat:** The `lang` field identifies the product language but does not guarantee that every ingredient phrase in the record is written in that language.

## Reported language counts

| Language | Product rows |
| -------- | -----------: |
| English  |    2,054,773 |
| French   |    1,386,936 |
| Spanish  |      385,414 |
| German   |      360,625 |
| Italian  |      256,815 |
| Dutch    |      103,694 |

These are historical scan results and may vary with a different dataset snapshot.

## English-first decision

The initial experiments use English-language records to establish a reproducible baseline.

This does not mean the final project is intended to be English-only.

Agreed direction:

1. Establish an English baseline.
2. Preserve original multilingual data and language metadata.
3. Expand to multilingual evaluation later.
4. Investigate Indian packaged foods and ingredient terminology.
5. Evaluate performance separately by language and product population.

Potential future models include mBERT and XLM-RoBERTa. These have not yet been trained or evaluated in this project.

Potential Indian ingredient terms include atta, maida, besan, paneer, ghee and masalas. NOVA classification must not be inferred from an ingredient name or country alone.

---

# 5. Dataset Preparation

## 5.1 `prepare_dataset.py`

Path:

```text
src/deep_learning/prepare_dataset.py
```

Purpose:

- Read the large Parquet file using PyArrow batches.
- Avoid loading the entire dataset into Pandas.
- Filter English records using `lang == "en"`.
- Skip missing or invalid NOVA labels.
- Skip records with missing ingredient text.
- Recursively extract ingredient text from structured data.
- Use reservoir sampling to select up to 2,500 products per NOVA class.
- Generate a balanced 10,000-product sample.

Implementation details:

- PyArrow Parquet batch processing.
- Batch size: 5,000.
- `use_threads=False`.
- Random seed: 42.
- Selected columns: `code`, `ingredients`, `nova_group`, `lang`.

Output:

```text
data/deep_learning/training_sample.csv
```

Columns:

```text
code, ingredient_text, nova_group
```

## 5.2 Full eligible English dataset counts

| NOVA group | Eligible English products |
| ---------- | ------------------------: |
| NOVA 1     |                    53,594 |
| NOVA 2     |                     8,684 |
| NOVA 3     |                    91,175 |
| NOVA 4     |                   387,061 |
| **Total**  |               **540,514** |

The balanced sample selected 2,500 records per class, giving 10,000 products.

---

# 6. Dataset Quality and Duplicate Investigation

## 6.1 `check_dataset.py`

Path:

```text
src/deep_learning/check_dataset.py
```

Reported sample checks:

| Check                                  |            Result |
| -------------------------------------- | ----------------: |
| Rows                                   |            10,000 |
| Columns                                |                 3 |
| Missing product codes                  |                 0 |
| Missing ingredient text                |                 0 |
| Missing NOVA labels                    |                 0 |
| Duplicate barcodes                     |                 0 |
| Initial exact repeated descriptions    |             2,315 |
| Unique descriptions                    |             7,685 |
| Descriptions shorter than 5 characters |               143 |
| Mean text length                       | 129.55 characters |
| Median text length                     |     56 characters |
| Maximum text length                    |  2,551 characters |

Each NOVA group had 2,500 products in the balanced sample.

Short descriptions such as `egg`, `salt`, `oil`, `honey` or `broccoli` can be valid single-ingredient products. They should not be deleted automatically.

## 6.2 `investigate_duplicates.py`

Path:

```text
src/deep_learning/investigate_duplicates.py
```

Normalization:

- Lowercase.
- Strip leading/trailing whitespace.
- Collapse repeated whitespace.

Reported results:

```text
Total products: 10000
Products with repeated descriptions: 3382
Repeated unique descriptions: 551
Descriptions with conflicting NOVA labels: 73
```

Examples of descriptions appearing with different NOVA labels include:

- `agave nectar`
- `almonds`
- `avocado`
- `avocado oil`
- `balsamic vinegar`
- `arrowroot starch`

Possible explanations include missing manufacturing context, label inconsistency and data-quality issues. These findings do not establish which label is correct.

**Decision:** Do not automatically delete all repeated descriptions or conflicting-label rows. Keep them documented and prevent identical normalized descriptions from crossing dataset partitions.

---

# 7. Train, Validation and Test Split

## 7.1 `split_dataset.py`

Path:

```text
src/deep_learning/split_dataset.py
```

The script created three partitions while ensuring normalized ingredient descriptions did not overlap across them.

Implementation:

- Normalize ingredient text.
- Use `GroupShuffleSplit`.
- Search 500 candidate splits.
- Use fixed random seeds 42 and 43.
- Target approximately 70% / 15% / 15%.

## Reported split sizes

| Partition  |   Products |
| ---------- | ---------: |
| Training   |      7,018 |
| Validation |      1,481 |
| Testing    |      1,501 |
| **Total**  | **10,000** |

## Class distribution

| NOVA group | Training | Validation | Test |
| ---------- | -------: | ---------: | ---: |
| NOVA 1     |    1,742 |        375 |  383 |
| NOVA 2     |    1,758 |        369 |  373 |
| NOVA 3     |    1,767 |        368 |  365 |
| NOVA 4     |    1,751 |        369 |  380 |

## Leakage verification

Reported output:

```text
train-validation 0
train-test 0
validation-test 0

SUCCESS: No ingredient-description leakage!
```

No normalized ingredient description was shared across the three partitions in the reported split.

**Important evaluation limitation:** This is a balanced test set. It is useful for comparing the initial models, but it does not represent the natural NOVA distribution of the full English dataset.

---

# 8. Text Quality and Long-Description Investigation

## 8.1 `analyze_text_quality.py`

Path:

```text
src/deep_learning/analyze_text_quality.py
```

Reported results from the 10,000-product sample:

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
| Short descriptions under 5 characters   |    143 |
| Long descriptions over 1,000 characters |     82 |
| Records containing numbers              |  1,699 |
| Records containing punctuation          |  7,297 |
| Records with non-ASCII characters       |    461 |
| Unusual whitespace                      |      0 |

Numbers, punctuation and non-ASCII characters were not automatically removed because they may carry useful ingredient information.

## 8.2 `analyze_full_dataset.py`

Path:

```text
src/deep_learning/analyze_full_dataset.py
```

The full Parquet file was scanned using PyArrow batches.

Reported results:

```text
Total rows: 4,895,629
Total English products: 2,054,773
Eligible English products: 540,514
Invalid or missing NOVA labels: 1,503,903
Missing ingredient text: 10,356
```

### Natural NOVA distribution in eligible English data

| NOVA group | Products |  Share |
| ---------- | -------: | -----: |
| NOVA 1     |   53,594 |  9.92% |
| NOVA 2     |    8,684 |  1.61% |
| NOVA 3     |   91,175 | 16.87% |
| NOVA 4     |  387,061 | 71.61% |

The natural dataset is strongly imbalanced, especially for NOVA 2 and NOVA 4.

### Full-dataset text lengths

| Measure                            |            Result |
| ---------------------------------- | ----------------: |
| Mean                               | 242.96 characters |
| Median                             |    179 characters |
| Minimum                            |       1 character |
| Maximum                            | 20,025 characters |
| 25th percentile                    |                84 |
| 75th percentile                    |               312 |
| Descriptions under 5 characters    |             2,877 |
| Descriptions over 1,000 characters |             9,330 |

### Long-description association

Of 9,330 descriptions longer than 1,000 characters:

| NOVA group | Long descriptions |
| ---------- | ----------------: |
| NOVA 1     |                 2 |
| NOVA 2     |                 0 |
| NOVA 3     |                19 |
| NOVA 4     |             9,309 |

Approximately 99.8% of these long descriptions belong to NOVA 4.

This is a strong association in the dataset, but it does not prove that long text causes NOVA 4 classification or that all long descriptions are poor quality.

## 8.3 `investigate_long_texts.py`

Path:

```text
src/deep_learning/investigate_long_texts.py
```

The script inspected the longest ingredient descriptions.

The longest reported description was 20,025 characters.

Some records contained genuine ingredients mixed with:

- Multilingual label text.
- Nutrition information.
- OCR noise.
- Multiple product varieties.
- Cooking and storage instructions.
- Warnings and manufacturer information.
- Packaging details.

## 8.4 `sample_long_texts.py`

Path:

```text
src/deep_learning/sample_long_texts.py
```

Output:

```text
data/deep_learning/long_text_review.csv
```

The script selected 30 descriptions longer than 1,000 characters using reservoir sampling and random seed 42.

All 30 selected descriptions belonged to NOVA 4, consistent with the full-dataset distribution.

**Decision:** Do not automatically remove all long descriptions. Preserve the source data and investigate defensible cleaning methods before changing the training dataset.

---

# 9. Generated Data Layout

Expected local data structure:

```text
data/
├── food.parquet
└── deep_learning/
    ├── training_sample.csv
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    ├── duplicate_analysis.csv
    ├── long_text_review.csv
    ├── cnn_best.pt
    ├── cnn_vocab.json
    ├── lstm_best.pt
    ├── lstm_vocab.json
    └── distilbert_food_nova/
```

The exact generated files should be checked in the local checkout.

The CSV, Parquet and model checkpoint files are local generated outputs and should remain excluded from Git unless the team explicitly decides otherwise.

---

# 10. Model Development and Experiments

Four experiments have now been completed:

1. TF-IDF + Logistic Regression.
2. 1D CNN.
3. Bidirectional LSTM.
4. DistilBERT Transformer.

All four models used the same training, validation and test partitions.

The test set contains 1,501 products.

This makes the initial model comparison consistent, although it does not replace evaluation on naturally distributed or external data.

## 10.1 Experiment 1 — TF-IDF + Logistic Regression

**Category:** Traditional Machine Learning  
**Status:** Completed

Script:

```text
src/deep_learning/train_baseline.py
```

### Model configuration

TF-IDF:

- Lowercase text.
- Unigrams and bigrams.
- Maximum features: 100,000.
- Minimum document frequency: 2.
- Sublinear term frequency enabled.

Logistic Regression:

- Maximum iterations: 1,000.
- Random state: 42.

### Results

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     82.58% | 82.01% |
| Macro F1 |     0.8282 | 0.8188 |

### Test classification report

| NOVA group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.86 |   0.94 |     0.90 |     383 |
| NOVA 2     |      0.92 |   0.62 |     0.74 |     373 |
| NOVA 3     |      0.64 |   0.84 |     0.72 |     365 |
| NOVA 4     |      0.95 |   0.88 |     0.91 |     380 |

### Confusion matrix

Rows are actual classes; columns are predicted classes.

```text
[[360,   4,  14,   5],
 [ 16, 230, 124,   3],
 [ 36,  13, 305,  11],
 [  7,   2,  35, 336]]
```

### Observations

- Test accuracy was 82.01%.
- NOVA 1 and NOVA 4 had stronger F1-scores.
- NOVA 2 was frequently misclassified as NOVA 3.
- Only 230 of 373 actual NOVA 2 products were correctly classified.

This model established the traditional ML baseline.

---

## 10.2 Experiment 2 — 1D CNN

**Category:** Deep Learning  
**Framework:** PyTorch  
**Device:** Apple MPS  
**Status:** Completed

Script:

```text
src/deep_learning/train_cnn.py
```

### Architecture

1. Ingredient text tokenization.
2. Vocabulary construction.
3. Embedding layer.
4. Three 1D convolution branches.
5. ReLU activation.
6. Global max pooling.
7. Dropout.
8. Fully connected output layer with four NOVA classes.

### Configuration

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

### Results

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     89.06% | 88.94% |
| Macro F1 |     0.8904 | 0.8889 |

### Test classification report

| NOVA group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.89 |   0.87 |     0.88 |     383 |
| NOVA 2     |      0.87 |   0.91 |     0.89 |     373 |
| NOVA 3     |      0.86 |   0.84 |     0.85 |     365 |
| NOVA 4     |      0.94 |   0.93 |     0.94 |     380 |

### Confusion matrix

```text
[[335,  28,  14,   6],
 [ 18, 340,  12,   3],
 [ 22,  23, 305,  15],
 [  2,   1,  22, 355]]
```

### Observations

- Test accuracy was 88.94%.
- The CNN correctly classified 1,335 out of 1,501 products.
- NOVA 4 achieved the strongest F1-score of 0.94.
- NOVA 3 remained the most challenging class.
- Training accuracy reached 97.45%, suggesting some overfitting compared with validation performance.
- Early stopping selected the best validation checkpoint.

The CNN outperformed the Logistic Regression baseline.

---

## 10.3 Experiment 3 — Bidirectional LSTM

**Category:** Deep Learning  
**Framework:** PyTorch  
**Device:** Apple MPS  
**Status:** Completed

Script:

```text
src/deep_learning/train_lstm.py
```

### Architecture

1. Ingredient text tokenization.
2. Vocabulary construction.
3. Embedding layer.
4. Forward LSTM.
5. Backward LSTM.
6. Combination of forward and backward representations.
7. Dropout.
8. Fully connected output layer with four classes.

The bidirectional architecture processes token sequences in both directions, allowing it to learn context from preceding and following words.

### Configuration

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

### Results

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     87.64% | 89.81% |
| Macro F1 |     0.8761 | 0.8977 |

### Test classification report

| NOVA group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.89 |   0.93 |     0.91 |     383 |
| NOVA 2     |      0.94 |   0.89 |     0.91 |     373 |
| NOVA 3     |      0.83 |   0.84 |     0.84 |     365 |
| NOVA 4     |      0.94 |   0.92 |     0.93 |     380 |

### Confusion matrix

```text
[[358,   5,  16,   4],
 [ 19, 333,  20,   1],
 [ 23,  17, 307,  18],
 [  3,   0,  27, 350]]
```

### Observations

- Test accuracy was 89.81%.
- The LSTM correctly classified 1,348 out of 1,501 products.
- NOVA 1 and NOVA 2 both achieved F1-scores of 0.91.
- NOVA 4 achieved an F1-score of 0.93.
- NOVA 3 remained the most challenging class, with an F1-score of 0.84.
- Training accuracy reached 97.89%, indicating a gap between training and validation performance.

The LSTM outperformed both the baseline and CNN on the test split.

---

## 10.4 Experiment 4 — DistilBERT Transformer

**Category:** Deep Learning  
**Framework:** PyTorch and Hugging Face Transformers  
**Device:** Apple MPS  
**Status:** Completed

Script:

```text
src/deep_learning/train_transformer.py
```

Saved model directory:

```text
data/deep_learning/distilbert_food_nova/
```

### Model architecture

Pretrained model:

```text
distilbert/distilbert-base-uncased
```

DistilBERT is a smaller pretrained Transformer derived from BERT.

The model uses self-attention to learn contextual relationships between tokens in ingredient descriptions.

The pretrained language model was fine-tuned for four-class NOVA classification.

Architecture:

1. DistilBERT tokenizer.
2. Token IDs and attention masks.
3. Pretrained DistilBERT encoder.
4. Classification head.
5. Four output classes representing NOVA 1–4.

The classification head was initialized for the new task and trained during fine-tuning.

### Configuration

| Setting                     |                   Value |
| --------------------------- | ----------------------: |
| Pretrained model            | DistilBERT base uncased |
| Maximum token length        |                     256 |
| Batch size                  |                       8 |
| Gradient accumulation steps |                       2 |
| Effective batch size        |                      16 |
| Learning rate               |          0.00002 (2e-5) |
| Optimizer                   |                   AdamW |
| Weight decay                |                    0.01 |
| Maximum epochs              |                       3 |
| Early stopping patience     |                       2 |
| Random seed                 |                      42 |

### Training progress

| Epoch | Training accuracy | Training macro F1 | Validation accuracy | Validation macro F1 |
| ----- | ----------------: | ----------------: | ------------------: | ------------------: |
| 1     |            80.01% |            0.8003 |              90.07% |              0.9003 |
| 2     |            90.71% |            0.9072 |              90.75% |              0.9071 |
| 3     |            93.10% |            0.9312 |              90.75% |              0.9078 |

The third epoch achieved the highest validation macro F1-score, and its checkpoint was saved for final testing.

Training accuracy being lower than validation accuracy in early epochs is possible because dropout is active during training and disabled during evaluation. Training metrics are also collected while the model weights are changing across batches.

### Final results

| Metric   | Validation |   Test |
| -------- | ---------: | -----: |
| Accuracy |     90.75% | 91.34% |
| Macro F1 |     0.9078 | 0.9132 |

### Test classification report

| NOVA group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.92 |   0.94 |     0.93 |     383 |
| NOVA 2     |      0.87 |   0.93 |     0.90 |     373 |
| NOVA 3     |      0.87 |   0.85 |     0.86 |     365 |
| NOVA 4     |      0.99 |   0.92 |     0.96 |     380 |

### Confusion matrix

```text
[[360,  15,   8,   0],
 [  9, 348,  16,   0],
 [ 19,  32, 312,   2],
 [  2,   6,  21, 351]]
```

### Observations

- Test accuracy was 91.34%.
- Test macro F1 was 0.9132.
- The model correctly classified 1,371 out of 1,501 products.
- NOVA 4 achieved precision of 0.99 and F1-score of 0.96.
- NOVA 1 achieved an F1-score of 0.93.
- NOVA 2 achieved an F1-score of 0.90.
- NOVA 3 remained the most challenging class, with an F1-score of 0.86.

DistilBERT achieved the strongest overall performance among the four experiments.

---

# 11. Final Model Comparison

All four models were evaluated using the same test set of 1,501 products.

| Model                        | Category          | Test accuracy | Test macro F1 |
| ---------------------------- | ----------------- | ------------: | ------------: |
| TF-IDF + Logistic Regression | Traditional ML    |        82.01% |        0.8188 |
| 1D CNN                       | Deep Learning     |        88.94% |        0.8889 |
| Bidirectional LSTM           | Deep Learning     |        89.81% |        0.8977 |
| **DistilBERT**               | **Deep Learning** |    **91.34%** |    **0.9132** |

## Model ranking

1. **DistilBERT:** 91.34% test accuracy.
2. **Bidirectional LSTM:** 89.81% test accuracy.
3. **1D CNN:** 88.94% test accuracy.
4. **TF-IDF + Logistic Regression:** 82.01% test accuracy.

DistilBERT improved test accuracy by:

- 9.33 percentage points over Logistic Regression.
- 2.40 percentage points over CNN.
- 1.53 percentage points over LSTM.

## Current model selection

**DistilBERT is the leading model candidate based on test accuracy and macro F1.**

However, this is not yet a final mobile deployment decision.

Before selecting a production model, evaluate:

- Model size.
- Inference time.
- Memory requirements.
- Android compatibility.
- Model conversion options.
- Performance on naturally distributed and external data.

A slightly less accurate model may be more practical for on-device inference.

---

# 12. Current Model Files and Inference Readiness

The training scripts save the following artifacts locally:

```text
data/deep_learning/
├── cnn_best.pt
├── cnn_vocab.json
├── lstm_best.pt
├── lstm_vocab.json
└── distilbert_food_nova/
```

The DistilBERT directory contains the saved model weights/configuration and tokenizer files produced by Hugging Face's `save_pretrained()` method.

The model has been saved successfully, but a standalone production inference interface has not yet been verified.

Next steps:

1. Build a reusable prediction script.
2. Load the saved tokenizer and model.
3. Accept ingredient text as input.
4. Return the predicted NOVA class.
5. Test on examples not used during training.
6. Measure inference latency and memory use.
7. Investigate model conversion and Android integration.

Do not claim the model is already integrated into the Android application.

---

# 13. Important Evaluation Limitations

The current model results are promising, but several limitations must be addressed before making real-world performance claims.

## 13.1 Balanced test distribution

The 10,000-product sample was deliberately balanced with 2,500 examples per NOVA class.

The full eligible English dataset has a very different distribution:

- NOVA 1: 9.92%.
- NOVA 2: 1.61%.
- NOVA 3: 16.87%.
- NOVA 4: 71.61%.

The current 91.34% DistilBERT test accuracy therefore describes performance on a balanced test set, not the natural product population.

A naturally distributed evaluation set should be created and evaluated separately.

## 13.2 Dataset label quality

The duplicate investigation found 73 normalized ingredient descriptions with conflicting NOVA labels.

This suggests that some labels may be inconsistent or may depend on product context not present in ingredient text.

The model learns from the supplied labels and cannot resolve missing manufacturing information automatically.

## 13.3 Ingredient-text limitations

Ingredient descriptions may include:

- OCR errors.
- Manufacturer information.
- Storage instructions.
- Nutrition information.
- Multiple product varieties.
- Multilingual text.
- Missing or incomplete ingredients.

These can affect model performance.

## 13.4 Long-text association

Approximately 99.8% of descriptions longer than 1,000 characters in the full eligible English dataset belonged to NOVA 4.

The model could potentially learn dataset-specific patterns associated with long descriptions.

Further experiments should investigate whether text length or irrelevant label content influences predictions.

## 13.5 Language limitation

All four current model experiments used the English-filtered balanced sample.

They do not establish performance on Hindi, other Indian languages or multilingual product descriptions.

---

# 14. Multilingual and India-Focused Future Direction

The project should retain its longer-term multilingual and India-focused goals.

## Phase A — English baseline

Completed:

- English-language dataset preparation.
- Balanced sample creation.
- Dataset splitting.
- Baseline and neural model experiments.
- Initial model comparison.

## Phase B — Multilingual expansion

Future tasks:

- Preserve original language metadata.
- Inspect actual text language quality.
- Evaluate multilingual model candidates such as mBERT and XLM-R.
- Report performance separately by language.
- Avoid claiming support for languages that have not been tested.

## Phase C — India-focused evaluation

Future tasks:

- Obtain a suitable labelled collection of Indian food products.
- Include Indian packaged foods and ingredient terminology.
- Evaluate Hindi and English first.
- Add regional languages where sufficient labelled data is available.
- Evaluate separately on Indian products.
- Continue to use NOVA classification criteria rather than assumptions based on cuisine or ingredient names.

These are future directions, not completed experiments.

---

# 15. Git Status and Workflow

The user previously reported a clean working tree on `feature/deep-learning` before the CNN work.

The user subsequently created and ran the CNN, LSTM and Transformer scripts and updated the experiment results document.

However, the latest Git status and commit history have not been provided after all four experiments.

**Therefore, the current remote commit status of these latest files is unknown.**

## Required Git verification

In VS Code → Integrated Terminal:

```bash
git status
```

Then inspect recent commits:

```bash
git log -5 --oneline
```

Stage only intended code and documentation files.

Expected relevant files include:

```text
src/deep_learning/train_cnn.py
src/deep_learning/train_lstm.py
src/deep_learning/train_transformer.py
docs/MODEL_EXPERIMENT_RESULTS.md
docs/DEEP_LEARNING_PROJECT_HANDOVER.md
```

The baseline script may already be tracked from earlier work; verify rather than assuming.

Do not commit:

- `.venv/`
- `data/`
- Generated CSV files.
- Parquet files.
- Large model checkpoints.
- Local cache files.
- `__pycache__/`

The saved model files are expected to remain local unless the team explicitly chooses a model-artifact sharing strategy.

## Git workflow

```bash
git status
```

Stage specific files:

```bash
git add src/deep_learning/train_transformer.py
git add docs/MODEL_EXPERIMENT_RESULTS.md
git add docs/DEEP_LEARNING_PROJECT_HANDOVER.md
```

Add other scripts only if they are still untracked or modified.

Commit:

```bash
git commit -m "Complete deep learning model experiments and documentation"
```

Push:

```bash
git push origin feature/deep-learning
```

Verify:

```bash
git status
```

Do not assume the push succeeded until Git confirms it.

---

# 16. Current Project Status

## Completed

- [x] Inspected the source Parquet structure.
- [x] Built memory-conscious PyArrow batch processing.
- [x] Extracted ingredient text recursively.
- [x] Created a balanced 10,000-product English sample.
- [x] Investigated missing values and duplicate barcodes.
- [x] Investigated repeated descriptions and conflicting NOVA labels.
- [x] Created train/validation/test splits.
- [x] Verified zero normalized ingredient-description overlap between partitions in the reported split.
- [x] Analyzed sample text quality.
- [x] Scanned the full English dataset.
- [x] Investigated long descriptions.
- [x] Created a reproducible long-description review sample.
- [x] Implemented TF-IDF + Logistic Regression baseline.
- [x] Trained and evaluated 1D CNN.
- [x] Trained and evaluated Bidirectional LSTM.
- [x] Fine-tuned and evaluated DistilBERT.
- [x] Compared all four models.
- [x] Identified DistilBERT as the leading model based on current test metrics.

## Remaining work

- [ ] Verify Git status and push all intended scripts and documentation.
- [ ] Verify the current handover and experiment-results documents are saved in the repository.
- [ ] Build a reusable inference/prediction script.
- [ ] Test predictions on new ingredient descriptions.
- [ ] Measure inference speed and model size.
- [ ] Evaluate on a naturally distributed dataset.
- [ ] Investigate the influence of long descriptions and dataset-specific patterns.
- [ ] Verify the BDA integrated Parquet output and schema.
- [ ] Decide on a model deployment strategy.
- [ ] Export or convert the selected model if required.
- [ ] Integrate the model with the Android application.
- [ ] Conduct multilingual evaluation.
- [ ] Conduct India-focused evaluation using suitable labelled data.

---

# 17. Immediate Next Steps

The recommended order is:

## Step 1 — Save and verify documentation

Ensure these documents are saved:

```text
docs/MODEL_EXPERIMENT_RESULTS.md
docs/DEEP_LEARNING_PROJECT_HANDOVER.md
```

The experiment-results document should contain all four experiments and their metrics.

## Step 2 — Verify Git

Run:

```bash
git status
git log -5 --oneline
```

Review the results before staging or committing anything.

## Step 3 — Build the prediction interface

Create a reusable inference script that:

- Loads the saved DistilBERT model.
- Loads the matching tokenizer.
- Accepts ingredient text.
- Returns one predicted NOVA group.
- Can later be called by an API or application layer.

## Step 4 — Test and benchmark

Evaluate:

- Prediction correctness on additional examples.
- Inference time.
- Model storage size.
- Memory use.
- Behaviour on empty, short and long ingredient descriptions.

## Step 5 — Prepare Android integration

Determine whether the model will run:

- Directly on the Android device after suitable conversion, or
- Through a backend inference API.

Do not assume the final approach until deployment requirements and app architecture are verified.

## Step 6 — Natural-distribution and multilingual evaluation

Create a representative evaluation set and assess performance under the natural class distribution.

Later, expand to multilingual and India-focused datasets.

---

# 18. Guidance for the Next Assistant or Collaborator

The user is a beginner in deep learning and prefers clear, step-by-step explanations.

Preferred working style:

1. Explain the immediate goal and why it matters.
2. Give clear instructions for VS Code Editor, Explorer and Integrated Terminal.
3. Explain technical terms in simple language.
4. Do not make the user repeat facts already documented here.
5. Avoid creating unnecessary scripts.
6. Distinguish completed work from planned work.
7. Do not claim Android integration is complete.
8. Do not promise a particular accuracy.
9. Do not automatically delete short, duplicate, conflicting or long descriptions.
10. Preserve the multilingual and India-focused direction.
11. Verify BDA integration outputs rather than assuming they exist.
12. Check Git status before staging or committing.
13. Keep generated datasets, model checkpoints and `.venv` out of Git unless explicitly agreed.
14. Explain the limitations of the balanced test set when discussing model performance.

## Exact stopping point

The four planned model experiments have been completed.

Current test leaderboard:

| Model               | Test accuracy | Test macro F1 |
| ------------------- | ------------: | ------------: |
| Logistic Regression |        82.01% |        0.8188 |
| 1D CNN              |        88.94% |        0.8889 |
| Bidirectional LSTM  |        89.81% |        0.8977 |
| **DistilBERT**      |    **91.34%** |    **0.9132** |

DistilBERT is the leading candidate based on the current balanced test set.

**The immediate next task is to verify Git and documentation, then build and test a reusable DistilBERT inference interface. After that, focus on deployment, natural-distribution evaluation and Android integration.**

---

**End of Deep Learning Handover**
