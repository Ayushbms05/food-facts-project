# Food Health Prediction Project — Deep Learning Handover

**Last updated: 8 November 2026**  
**Workstream: Member 2 — Deep Learning / Machine Learning**

This is the current, comprehensive handover for the Deep Learning (DL) workstream of the Food Health Prediction project. It is intended for the project owner, teammates, and a future ChatGPT conversation so work can continue without repeating completed investigations.

> **Status discipline:** This document distinguishes confirmed results from plans and items that still need verification. Do not assume that a file was committed, pushed, or shared unless the user confirms it or a fresh Git status proves it.

---

# 1. Project overview

The wider Food Health Prediction project has three workstreams:

1. **Big Data Analytics (BDA):** process and integrate large food-product datasets, including Open Food Facts (OFF) and USDA data, using Apache Spark.
2. **Deep Learning / Machine Learning:** predict a product's NOVA food-processing group (1–4) from its ingredient text.
3. **Mobile application:** Android-first app is being considered; Flutter has been discussed as a possible technology.

The current DL research question is:

> **Can a model predict a product's NOVA group from its ingredient text?**

## What NOVA means

NOVA classifies foods according to the nature and extent of processing. It is **not simply a healthy-versus-unhealthy label**.

- **NOVA 1:** unprocessed or minimally processed foods.
- **NOVA 2:** processed culinary ingredients.
- **NOVA 3:** processed foods.
- **NOVA 4:** ultra-processed foods.

The exact classification can depend on ingredients, manufacturing processes, and product context. Ingredient text alone may not contain all the information needed to assign a NOVA group. Identical or near-identical ingredient descriptions can appear with different NOVA labels.

## Important project limitation

A model trained only on ingredient text can learn patterns associated with the labels, but it cannot reliably infer manufacturing details that are absent from the text. Dataset labels may also contain inconsistencies. Model results must be presented with these limitations.

---

# 2. Team and workstream context

Known project roles and context:

- **BDA workstream:** Apache Spark processing and integration of OFF and USDA data.
- **DL workstream:** ingredient-text extraction, dataset preparation, quality investigation, and model development.
- **Mobile app workstream:** Android-first application; Flutter has been discussed.

## Historical BDA integration results

The following figures were reported in shared project conversations by the BDA workstream. They have not been independently re-verified as part of this DL handover.

| Metric | Reported value |
|---|---:|
| Matched products | 368,055 |
| Products with OFF ingredients | 314,306 |
| Products with USDA ingredients | 365,741 |
| Products with both ingredient sources | 313,501 |
| Extra OFF records | 1 |

Historical output paths mentioned:

```text
data/integration/combined_products.parquet
data/integration/usda_latest.parquet
```

**Do not assume these paths exist in the DL user's local checkout.** Verify the files, schema, provenance, and branch before changing the DL source dataset. The DL work described below has been performed using `data/food.parquet`, not the confirmed integrated BDA output.

The BDA teammate also reported a corrected lookup summary:

| Lookup category | Count |
|---|---:|
| Found with NOVA | 230 |
| Found without NOVA | 40 |
| Sentinel barcode collision | 30 |

These are teammate-reported results, not a DL-side verification.

---

# 3. Repository, branch, and local environment

## Repository and branch

Repository folder:

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

Develop and push DL changes to `feature/deep-learning`, not directly to `main`.

## Observed project structure

The following is the expected/observed layout. Some files and generated outputs are local and may be ignored by Git.

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
│       └── sample_long_texts.py
├── docs/
├── notebooks/
├── notes/
├── data/
├── .venv/
├── .gitignore
└── README.md
```

A baseline training script has been proposed but **has not been confirmed as created or run**:

```text
src/deep_learning/train_baseline.py
```

## Local machine and software

Known local setup:

- MacBook Air with Apple M1 chip.
- 8 GB unified memory.
- Python 3.12.5, ARM64.
- macOS 26.3 was reported.
- Existing project virtual environment: `.venv`.
- Keep using the existing `.venv`; do not recreate it without a reason.
- Terminal prompt has shown both `(.venv)` and `(base)`. Confirm the project virtual environment is active when running scripts.

Previously reported package versions/status:

| Library / capability | Reported version or status |
|---|---|
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| PyArrow | 26.0.1 |
| PyTorch | 2.15.0 |
| PyTorch MPS build | Available |
| PyTorch MPS device | Available |

The environment check previously reported:

```python
torch.backends.mps.is_built()      # True
torch.backends.mps.is_available()  # True
```

The device string was `mps`.

### Apple Metal / MPS explanation

Apple Metal is Apple's GPU-computing technology. PyTorch exposes GPU support on Apple silicon through the **MPS (Metal Performance Shaders) backend**. This can accelerate supported neural-network operations on the M1 GPU without an NVIDIA GPU.

Example device selection:

```python
import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using:", device)
```

For PyTorch models and tensors, the model and tensors must be moved to the same device. MPS operation coverage and performance vary by model and operation; CPU fallback may be needed for unsupported operations.

### Hardware constraint

The Mac has 8 GB unified memory shared by CPU and GPU. Use small batches, modest sequence lengths, and compact models. Do not assume a large transformer can be trained comfortably locally. A cloud GPU may be useful for larger experiments.

### Memory-safe data handling

The source Parquet file contains almost 4.9 million rows. Do **not** load the entire Parquet dataset into Pandas. The full-dataset scripts use PyArrow batch iteration (5,000 rows per batch) and process rows incrementally.

---

# 4. Source dataset and characteristics

Current DL source file:

```text
data/food.parquet
```

Reported Parquet row count:

```text
4,895,629
```

Relevant columns:

- `code`: product barcode/code.
- `ingredients`: structured JSON-like ingredient data, not plain text.
- `nova_group`: NOVA class, expected values 1–4 plus missing/invalid values.
- `lang`: product language code.

Ingredient objects may contain a `text` field and nested `ingredients` lists/objects. The extraction function recursively extracts both parent and nested ingredient text.

**Language caveat:** `lang` is a product language field. It does not guarantee every ingredient phrase in the record is in that language.

## Previously reported language counts

| Language | Reported product rows |
|---|---:|
| English (`en`) | 2,054,773 |
| French (`fr`) | 1,386,936 |
| Spanish (`es`) | 385,414 |
| German (`de`) | 360,625 |
| Italian (`it`) | 256,815 |
| Dutch (`nl`) | 103,694 |

Other languages are present. These are historical scan results and may change with a different dataset version.

## English-first decision and future multilingual scope

The first sample was filtered using `lang == "en"` to establish a manageable baseline. This is a practical starting point, **not a decision to make the final project English-only**.

Agreed direction:

1. Establish a reproducible English baseline first.
2. Preserve the original multilingual Parquet source and language information.
3. Avoid designing the entire pipeline around an irreversible English-only assumption.
4. Later evaluate multilingual approaches using suitable multilingual text models and language-specific metrics.
5. Develop an India-focused evaluation/expansion using Indian packaged products, ingredient terms, and languages where sufficient labelled data is available.

Potential future multilingual models include mBERT and XLM-RoBERTa. They are future candidates, not models already trained or evaluated in this project.

Potential India-focused terms and product context include atta, maida, besan, paneer, ghee, masalas, regional packaged foods, and Indian product brands. Do not infer NOVA class from an ingredient name or country alone; continue to apply NOVA criteria.

---

# 5. Dataset preparation

## 5.1 `prepare_dataset.py`

Path:

```text
src/deep_learning/prepare_dataset.py
```

Purpose:

- Read `data/food.parquet` using PyArrow batches.
- Avoid loading the full dataset into Pandas.
- Keep English product records for the initial baseline (`lang == "en"`).
- Skip records with missing/invalid NOVA labels or empty ingredient text.
- Recursively extract text from the structured ingredient field, including nested ingredients.
- Use reservoir sampling to select up to 2,500 products per NOVA class.
- Save a balanced 10,000-row sample.

Implementation details reported:

- `pyarrow.parquet.ParquetFile`
- `iter_batches(batch_size=5000, ...)`
- Selected columns: `code`, `ingredients`, `nova_group`, `lang`
- `use_threads=False`
- Random seed: 42
- Output: `data/deep_learning/training_sample.csv`
- Columns: `code`, `ingredient_text`, `nova_group`

### Full eligible English counts from the preparation/full-scan work

| NOVA group | Eligible English records |
|---|---:|
| 1 | 53,594 |
| 2 | 8,684 |
| 3 | 91,175 |
| 4 | 387,061 |
| **Total** | **540,514** |

The script selected 2,500 records from each group, for 10,000 total.

An earlier non-English-filtered run reported different eligible counts:

| NOVA group | Earlier all-language eligible count |
|---|---:|
| 1 | 133,848 |
| 2 | 20,672 |
| 3 | 217,683 |
| 4 | 755,503 |

The English-filtered run overwrote `training_sample.csv`. The earlier all-language sample was not confirmed as separately saved.

---

# 6. Dataset quality checks and duplicate investigation

## 6.1 `check_dataset.py`

Path:

```text
src/deep_learning/check_dataset.py
```

Purpose: inspect the 10,000-row sample for missing values, duplicate product codes, repeated descriptions, NOVA distribution, and text length.

Reported results:

| Check | Result |
|---|---:|
| Rows | 10,000 |
| Columns | 3 |
| Missing `code` | 0 |
| Missing `ingredient_text` | 0 |
| Missing `nova_group` | 0 |
| Duplicate barcodes | 0 |
| Exact repeated ingredient descriptions (initial check) | 2,315 |
| Unique descriptions | 7,685 |
| Descriptions shorter than 5 characters | 143 |
| Mean text length | 129.55 characters |
| Median text length | 56 characters |
| Maximum text length | 2,551 characters |

NOVA sample distribution:

| NOVA group | Sample count |
|---|---:|
| 1 | 2,500 |
| 2 | 2,500 |
| 3 | 2,500 |
| 4 | 2,500 |

Short ingredient descriptions should not be removed solely because they are short. Examples such as `egg`, `salt`, `oil`, `honey`, or `broccoli` can be valid descriptions of single-ingredient products.

Different barcodes with the same ingredient description are not necessarily duplicate products.

## 6.2 `investigate_duplicates.py`

Path:

```text
src/deep_learning/investigate_duplicates.py
```

Purpose: normalize ingredient descriptions and investigate whether the same description occurs with different NOVA labels.

Normalization reported:

- lowercase
- trim leading/trailing whitespace
- collapse repeated whitespace

Latest reported results:

```text
Total products: 10000
Products with repeated descriptions: 3382
Repeated unique descriptions: 551
Descriptions with conflicting NOVA labels: 73
```

Examples:

| Normalized ingredient description | NOVA labels observed |
|---|---|
| `_peanuts_, sea salt` | 2, 3 |
| `agave nectar` | 1, 2 |
| `almonds` | 1, 2 |
| `apple cider vinegar, diluted with water to 5% acidity` | 1, 2 |
| `apple cider vinegar, filtered water` | 1, 2 |
| `arrowroot starch` | 2, 3 |
| `avocado` | 1, 2 |
| `avocado oil` | 2, 3 |
| `balsamic vinegar` | 1, 2 |

Interpretation:

- 3,382 product rows belong to repeated descriptions.
- Those rows correspond to 551 distinct normalized descriptions that repeat.
- 73 repeated descriptions have more than one NOVA label.
- Potential explanations include missing product/manufacturing context, label inconsistency, and data-quality issues. These results do not prove which label is correct.

**Decision:** Do not automatically delete all repeated descriptions or all conflicting-label rows. Keep the data for now, document the limitation, and prevent identical normalized descriptions from leaking across train/validation/test partitions.

An earlier version of the script may have written:

```text
data/deep_learning/duplicate_analysis.csv
```

Verify the current script before relying on that generated file.

---

# 7. Train / validation / test split

## 7.1 `split_dataset.py`

Path:

```text
src/deep_learning/split_dataset.py
```

**Status: implemented and run successfully according to the user's reported output.**

Purpose:

- Split the 10,000-row balanced sample into training, validation, and test sets.
- Keep identical normalized ingredient descriptions in the same partition.
- Reduce data leakage risk.
- Aim for approximately 70% / 15% / 15% split proportions while preserving class balance.

Implementation reported:

- Normalize ingredient text (lowercase, trim, collapse whitespace).
- Use `GroupShuffleSplit`.
- Search 500 candidate splits for a good balance of class proportions and split sizes.
- Fixed seeds 42 and 43.
- Save `train.csv`, `validation.csv`, and `test.csv` under `data/deep_learning/`.

### Latest reported split sizes

| Partition | Rows | Approx. share |
|---|---:|---:|
| Training | 7,018 | 70.18% |
| Validation | 1,481 | 14.81% |
| Test | 1,501 | 15.01% |
| **Total** | **10,000** | **100%** |

### NOVA distribution by split

| NOVA group | Training | Validation | Test |
|---|---:|---:|---:|
| 1 | 1,742 | 375 | 383 |
| 2 | 1,758 | 369 | 373 |
| 3 | 1,767 | 368 | 365 |
| 4 | 1,751 | 369 | 380 |

The reported percentages in each split were close to 25% per class.

### Leakage verification

```text
train-validation 0
train-test 0
validation-test 0
SUCCESS: No ingredient-description leakage!
```

This is an important milestone: no normalized ingredient description was shared across the three partitions in that reported run.

### Important evaluation caveat

This is a **balanced test set created from a balanced sample**. It is useful for comparing initial models across the four classes, but it does not represent the natural NOVA distribution in the full English dataset. Later, evaluate on a representative/natural-distribution sample as well, or clearly state this limitation in reports.

---

# 8. Text quality investigation

## 8.1 `analyze_text_quality.py`

Path:

```text
src/deep_learning/analyze_text_quality.py
```

Purpose: examine the sampled ingredient text for length, unusual content, numbers, punctuation, non-ASCII characters, and possible contamination.

Reported results for the 10,000-row sample:

| Measure | Result |
|---|---:|
| Number of records | 10,000 |
| Mean characters | 129.55 |
| Standard deviation | 196.81 |
| Minimum characters | 3 |
| 25th percentile | 17 |
| Median characters | 56 |
| 75th percentile | 160 |
| Maximum characters | 2,551 |
| Mean words | 17.79 |
| Median words | 8 |
| Maximum words | 361 |
| Short descriptions under 5 characters | 143 |
| Long descriptions over 1,000 characters | 82 |
| Records containing numbers | 1,699 (16.99%) |
| Records containing punctuation | 7,297 (72.97%) |
| Records with non-ASCII characters | 461 (4.61%) |
| Unusual whitespace | 0 |

Examples of short descriptions included `Pork`, `Eggs`, `oats`, `Corn`, `SALT`, `Beef`, and `milk`. These can be valid and should not be removed automatically.

Some long descriptions included ingredient lists mixed with label text, warnings, packaging/recycling instructions, manufacturer details, marketing text, or non-English content.

**Decision:** Do not indiscriminately strip numbers, punctuation, or non-ASCII characters. They can carry useful information such as ingredient percentages, quantities, additive codes, and language-specific words. Inspect and clean based on evidence.

---

# 9. Full-dataset analysis

## 9.1 `analyze_full_dataset.py`

Path:

```text
src/deep_learning/analyze_full_dataset.py
```

Purpose: scan the full Parquet dataset in batches to count eligible English products, NOVA distribution, ingredient-text lengths, and long descriptions by NOVA group.

The script uses:

- `pyarrow.parquet.ParquetFile`
- `iter_batches(batch_size=5000, columns=[...], use_threads=False)`
- recursive ingredient-text extraction
- `lang == "en"` filter
- valid NOVA values 1–4
- non-empty ingredient text requirement

### Latest full-dataset scan output (8 November 2026)

```text
Starting full dataset analysis...
Dataset: data/food.parquet
Total rows in Parquet file: 4,895,629

Processed approximately 500,000 rows...
Processed approximately 1,000,000 rows...
Processed approximately 1,500,000 rows...
Processed approximately 2,000,000 rows...
Processed approximately 2,500,000 rows...
Processed approximately 3,000,000 rows...
Processed approximately 3,500,000 rows...
Processed approximately 4,000,000 rows...
Processed approximately 4,500,000 rows...

========== FULL DATASET ANALYSIS ==========
Total English products: 2,054,773
Eligible English products: 540,514
Invalid or missing NOVA labels: 1,503,903
Products with missing ingredient text: 10,356

========== NOVA DISTRIBUTION ==========
NOVA 1: 53,594 products (9.92%)
NOVA 2: 8,684 products (1.61%)
NOVA 3: 91,175 products (16.87%)
NOVA 4: 387,061 products (71.61%)

========== LONG DESCRIPTIONS BY NOVA ==========
NOVA 1: 2 long descriptions (0.00% of that NOVA group)
NOVA 2: 0 long descriptions (0.00% of that NOVA group)
NOVA 3: 19 long descriptions (0.02% of that NOVA group)
NOVA 4: 9,309 long descriptions (2.41% of that NOVA group)
Total long descriptions: 9,330

========== INGREDIENT TEXT LENGTH ==========
Average characters: 242.96
Median characters: 179.00
Minimum characters: 1
Maximum characters: 20,025
25th percentile: 84.00
75th percentile: 312.00

Descriptions shorter than 5 characters: 2,877
Descriptions longer than 1000 characters: 9,330
Percentage of long descriptions: 1.73%

Analysis complete.
```

### Full-dataset NOVA distribution

| NOVA group | Eligible English products | Share |
|---|---:|---:|
| 1 | 53,594 | 9.92% |
| 2 | 8,684 | 1.61% |
| 3 | 91,175 | 16.87% |
| 4 | 387,061 | 71.61% |
| **Total** | **540,514** | **100% (rounding)** |

The natural English eligible population is strongly imbalanced, with NOVA 4 representing about 71.61% and NOVA 2 only about 1.61%.

### Full-dataset ingredient-text lengths

| Measure | Result |
|---|---:|
| Mean | 242.96 characters |
| Median | 179 characters |
| Minimum | 1 character |
| Maximum | 20,025 characters |
| 25th percentile | 84 characters |
| 75th percentile | 312 characters |
| Under 5 characters | 2,877 |
| Over 1,000 characters | 9,330 (1.73%) |

### Important long-description finding

Of the 9,330 eligible English descriptions longer than 1,000 characters:

- NOVA 1: 2
- NOVA 2: 0
- NOVA 3: 19
- NOVA 4: 9,309

Thus 9,309 / 9,330 (about 99.8%) of the long descriptions are NOVA 4. Within NOVA 4, 2.41% of its eligible products have descriptions over 1,000 characters. The other classes have extremely few such descriptions.

This is a strong class association in this dataset. It does **not** prove that long text causes NOVA 4 classification, nor that all long descriptions are poor quality. It may reflect the dataset's product mix and how product labels are recorded.

---

# 10. Long-description investigation

## 10.1 `investigate_long_texts.py`

Path:

```text
src/deep_learning/investigate_long_texts.py
```

Purpose: find and inspect the 20 longest ingredient descriptions in the full eligible English dataset.

The script initially had a heap tie-comparison issue when two records had equal lengths. It was fixed by adding a unique counter as a tie-breaker:

```python
record_counter += 1
heap_item = (record["length"], record_counter, record)
```

Latest reported findings:

- Maximum description length: 20,025 characters.
- Long records often contain genuine ingredient data mixed with other label information.
- Examples included multilingual label text (including Russian), nutrition values, OCR noise, multiple product varieties, instructions, warnings, manufacturer information, and packaging text.

Examples among the longest records:

| Rank | Product code | NOVA | Text length |
|---|---|---:|---:|
| 1 | `4607084352115` | 4 | 20,025 |
| 2 | `7622202013584` | 4 | 11,523 |
| 3 | `4009176483262` | 4 | 8,563 |
| 4 | `8711327624764` | 4 | 8,172 |

These are investigation examples, not necessarily representative of all long descriptions.

## 10.2 `sample_long_texts.py`

Path:

```text
src/deep_learning/sample_long_texts.py
```

Purpose: use reproducible reservoir sampling to select 30 descriptions longer than 1,000 characters from the full eligible English population and save them for manual review.

Output:

```text
data/deep_learning/long_text_review.csv
```

Columns:

- `code`
- `nova_group`
- `text_length`
- `ingredient_text`

Latest reported run:

```text
Total long descriptions found: 9,330
Random descriptions selected: 30
Random seed: 42
Saved to: data/deep_learning/long_text_review.csv

Sample NOVA distribution:
NOVA 1: 0
NOVA 2: 0
NOVA 3: 0
NOVA 4: 30
```

All 30 sampled descriptions happened to be NOVA 4. The full-dataset analysis later confirmed that 9,309 of the 9,330 long descriptions are NOVA 4, so the sample result is consistent with the full scan.

Manual inspection of the sample found:
- multiple product descriptions/varieties combined;
- multilingual ingredient and label text;
- storage and cooking instructions;
- nutrition/health/supplement information;
- manufacturer addresses, phone numbers and websites;
- genuine complex ingredient lists.

### Decision on long descriptions

**Do not automatically delete all descriptions over 1,000 characters.**

Reasons:
- Some contain useful and genuine ingredient information.
- Deleting all of them would disproportionately remove NOVA 4 examples.
- The long-text association with NOVA 4 is a potential shortcut/leakage-like signal the model might exploit, but it is not enough by itself to justify deleting the records.

Current plan:
1. Preserve the original source records.
2. Do not delete these rows from the dataset at this stage.
3. Investigate and clean clearly irrelevant label content if a defensible method is developed.
4. Compare performance with and without long descriptions or with a carefully defined text-cleaning treatment.
5. Report the effect on each class, especially NOVA 4.

---

# 11. Current generated data layout

Expected local/generated files:

```text
data/
├── food.parquet
└── deep_learning/
    ├── training_sample.csv
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    ├── long_text_review.csv
    └── duplicate_analysis.csv        # may depend on script version
```

These generated datasets are local outputs and are intended to remain outside Git. They may need to be regenerated on another machine, provided the original Parquet source is obtained separately.

---

# 12. Current Git status and workflow

## Last status actually shown by the user

The latest Git status shared in the conversation was:

```text
On branch feature/deep-learning
Your branch is up to date with 'origin/feature/deep-learning'.

Changes not staged for commit:
  modified:   src/deep_learning/analyze_full_dataset.py

Untracked files:
  src/deep_learning/investigate_long_texts.py
  src/deep_learning/sample_long_texts.py

no changes added to commit
```

At that point, the following three scripts needed staging/commit/push:

```text
src/deep_learning/analyze_full_dataset.py
src/deep_learning/investigate_long_texts.py
src/deep_learning/sample_long_texts.py
```

The assistant advised:

```bash
git add src/deep_learning/analyze_full_dataset.py \
        src/deep_learning/investigate_long_texts.py \
        src/deep_learning/sample_long_texts.py

git commit -m "Add long ingredient text analysis"

git push origin feature/deep-learning

git status
```

**The user did not confirm the result of those commands before continuing.** Therefore, do not assume these files were pushed. Check a fresh `git status` and `git log` before making Git claims.

The handover document itself was then uploaded in the conversation for revision. Its Git status is unknown.

## Git safety rules

- Check `git status` before staging.
- Stage only specific code and documentation files.
- Do not use `git add .` without reviewing what it will stage.
- Do not commit `.venv/`, `data/`, Parquet files, generated CSVs, `__pycache__/`, or large local outputs.
- Do not assume a commit or push succeeded without seeing confirmation.

Intended `.gitignore` content previously discussed:

```gitignore
.venv/
venv/
data/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.csv
*.parquet
```

Note that broad `*.csv` and `*.parquet` rules ignore these file types throughout the repository. If the team later wants to track a small example dataset, review the rules and make a deliberate exception.

## Recommended branch-sync workflow

When the working tree is clean and local work is committed:

```bash
git switch feature/deep-learning
git fetch origin
git merge origin/main
git push origin feature/deep-learning
```

Merge `main` into the feature branch at sensible milestones. Do not push DL work directly to `main`.

---

# 13. Model development plan

The project has not yet confirmed a trained baseline model. The next modelling stage is to build and evaluate models in a controlled sequence.

## 13.1 First baseline: TF-IDF + Logistic Regression

A baseline training script was proposed:

```text
src/deep_learning/train_baseline.py
```

It has **not been confirmed as created or executed**.

Proposed pipeline:

1. Load `train.csv`, `validation.csv`, and `test.csv`.
2. Convert ingredient text into numerical features using `TfidfVectorizer`.
3. Use word unigrams and bigrams (`ngram_range=(1, 2)`) as an initial feature setup.
4. Train `LogisticRegression`.
5. Evaluate on validation and test partitions.
6. Report accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix.

Why start with this if the workstream is called Deep Learning?
- It gives us a fast, understandable benchmark.
- It shows whether ingredient text contains learnable signal.
- It provides a reference point for judging whether CNNs, LSTMs, or transformers genuinely improve performance.

The baseline is traditional machine learning, not a neural network. It is an important comparison model.

## 13.2 Planned CNN experiment

A 1D Convolutional Neural Network (CNN) can learn local patterns across sequences of ingredient tokens. It is a planned neural-network experiment, not yet implemented or trained.

Possible high-level pipeline:
- tokenize ingredient descriptions;
- convert tokens to integer IDs;
- use an embedding layer;
- apply one-dimensional convolution and pooling;
- classify into four NOVA classes.

## 13.3 Planned LSTM experiment

An LSTM (Long Short-Term Memory network) processes token sequences and can learn sequential relationships across ingredient text. It is planned, not yet implemented or trained.

Possible high-level pipeline:
- tokenize text;
- map tokens to IDs;
- use embeddings;
- process sequences with an LSTM;
- use a classifier head for NOVA 1–4.

## 13.4 Planned transformer experiment

A compact transformer-based text classifier may be evaluated after baseline and smaller neural models. Candidate models may include DistilBERT for English. For multilingual work, candidates include mBERT and XLM-RoBERTa.

These are candidates only. Do not claim that any has been fine-tuned or evaluated yet.

## 13.5 Agreed model progression

| Order | Model | Status |
|---|---|---|
| 1 | TF-IDF + Logistic Regression | Next; proposed, not yet run |
| 2 | 1D CNN | Planned |
| 3 | LSTM | Planned |
| 4 | Transformer (e.g. compact English model) | Planned |
| 5 | Multilingual transformer (mBERT/XLM-R) | Future expansion |

Not every architecture must be used in the final mobile app. Compare performance, class-wise reliability, model size, inference speed, and device constraints before selecting a final model.

## 13.6 Evaluation metrics

Do not report accuracy alone.

Use:
- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** among predictions assigned to a class, how many are correct.
- **Recall:** among true examples of a class, how many the model finds.
- **Macro-F1:** F1 calculated per class and averaged equally across the four classes.
- **Confusion matrix:** shows which classes are mistaken for which others.

Because the current sample and test split are balanced artificially, balanced-test accuracy is useful for controlled comparison but does not represent real-world performance on the natural population. Later create/evaluate on a representative naturally distributed test set and report both types of results clearly.

---

# 14. Multilingual and India-focused future direction

The user explicitly wants the project to expand beyond English and potentially specialize for India.

Agreed strategy:

### Phase A — English baseline first
- Keep current initial training English-filtered.
- Establish baseline performance and evaluation.
- Avoid adding multilingual complexity before the first reproducible experiment.

### Phase B — multilingual expansion
- Preserve original language metadata.
- Inspect actual text language quality rather than assuming `lang == "en"` guarantees English text.
- Evaluate multilingual model candidates such as mBERT or XLM-R.
- Report performance separately by language as well as overall.
- Do not claim support for a language until tested on suitable labelled data.

### Phase C — India-focused evaluation and specialization
- Build/obtain a suitable labelled collection of Indian food products.
- Include Indian packaged foods and ingredient terminology.
- Consider Hindi and English first, then regional languages where sufficient quality data is available.
- Evaluate separately on Indian products; do not assume performance transfers from international products.
- Continue to classify using NOVA methodology, not assumptions based on cuisine, brand, or ingredient names.

**Current decision:** Do not change the first English baseline to multilingual right now. Preserve the broader future direction in the pipeline design and documentation.

---

# 15. Current status: completed, in progress, and not yet done

## Completed and reported

- [x] Inspected source Parquet structure and relevant columns.
- [x] Confirmed the source has approximately 4.9 million rows.
- [x] Built memory-conscious PyArrow batch processing.
- [x] Extracted ingredient text recursively from structured JSON.
- [x] Created a balanced 10,000-row English sample (2,500 per NOVA class).
- [x] Checked missing values, duplicate barcodes, repeated descriptions, and text lengths.
- [x] Investigated repeated ingredient descriptions and conflicting NOVA labels.
- [x] Created train/validation/test partitions with normalized-text group isolation.
- [x] Verified zero normalized-description overlap between partitions in the reported split.
- [x] Analyzed text quality in the sample.
- [x] Scanned the full dataset for English eligible counts and NOVA distribution.
- [x] Investigated the longest descriptions.
- [x] Created a reproducible 30-record long-description review sample.
- [x] Counted long descriptions by NOVA group across the full eligible English dataset.
- [x] Discussed Apple MPS support and confirmed it was available in the earlier environment check.
- [x] Agreed on an English-first but multilingual/India-aware long-term direction.
- [x] Agreed that CNN, LSTM, and transformer experiments belong in the planned DL workstream.

## In progress / needs Git verification

- [ ] Verify whether the three long-description scripts were committed and pushed.
- [ ] Verify whether this revised handover has been saved into the repository and pushed.
- [ ] Check fresh Git status and branch state before the next Git operation.

## Not yet completed

- [ ] Create and run `train_baseline.py`.
- [ ] Interpret baseline validation and test results.
- [ ] Create a representative natural-distribution evaluation set.
- [ ] Decide and test defensible text-cleaning rules.
- [ ] Implement and train a CNN.
- [ ] Implement and train an LSTM.
- [ ] Evaluate a transformer model.
- [ ] Compare model metrics and resource requirements.
- [ ] Verify BDA integrated Parquet availability/schema and decide whether it should become a future DL source.
- [ ] Conduct multilingual and India-focused data collection/evaluation.
- [ ] Integrate the selected model with the mobile application.

---

# 16. Immediate next steps

The exact next steps should be done in this order.

## Step 1 — Git checkpoint
In **VS Code → Integrated Terminal**:

```bash
git status
```

Review the output. If the three long-text scripts are still uncommitted, stage only those scripts. Also stage this revised handover after it has been placed in the intended repository path, for example:

```text
docs/DEEP_LEARNING_PROJECT_HANDOVER.md
```

Do not stage generated CSV or Parquet files.

## Step 2 — Build the first baseline
Create:

```text
src/deep_learning/train_baseline.py
```

Use the existing `train.csv`, `validation.csv`, and `test.csv`. Start with TF-IDF + Logistic Regression. Do not regenerate the splits unless the data or split logic changes.

## Step 3 — Run and inspect results
In **VS Code → Integrated Terminal**:

```bash
python src/deep_learning/train_baseline.py
```

Share the complete output and review:
- accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- confusion matrix.

Do not assume a target accuracy in advance. Interpret the model's strengths and weaknesses based on the output.

## Step 4 — Record and commit the baseline milestone
Once the script runs and its results are understood:
- document the experiment and results;
- stage the Python script and relevant documentation only;
- commit and push to `feature/deep-learning`.

After the baseline is stable, proceed to a small CNN experiment, then LSTM, then a transformer comparison.

---

# 17. Guidance for the next assistant / collaborator

The user is a beginner in deep learning and wants to understand the project, not blindly copy code. Use clear, patient explanations without being patronizing.

Preferred working style:

1. Explain the immediate goal and why it matters.
2. Give a coherent sequence of steps, but pause at meaningful checkpoints when the next action depends on terminal output.
3. Label locations clearly:
   - **VS Code → Editor** for creating or editing code.
   - **VS Code → Explorer** for locating files.
   - **VS Code → Integrated Terminal** for commands.
4. Explain technical terms in simple language.
5. Do not make the user repeat facts already present in this handover.
6. Do not create extra scripts unnecessarily; reuse existing scripts when practical.
7. Clearly distinguish completed work from proposed future work.
8. Do not claim the model has been trained; baseline training has not yet been confirmed.
9. Do not promise a particular accuracy such as 80–90%.
10. Do not automatically delete short descriptions, duplicate descriptions, conflicting labels, or long descriptions.
11. Preserve language metadata and the future multilingual/India direction.
12. Verify the BDA integrated dataset rather than assuming it exists locally.
13. Track Git commits carefully and never assume a push succeeded without confirmation.
14. Keep generated datasets, `.venv`, and large data files out of Git.

## Exact stopping point

The user has completed the dataset investigation and is ready to begin model development. The last major decision was to proceed with:

1. TF-IDF + Logistic Regression baseline;
2. CNN;
3. LSTM;
4. Transformer;
5. multilingual transformer evaluation later, with an India-focused direction.

The baseline script code was provided in the conversation, but the user has not confirmed creating or running `train_baseline.py`.

**Start by checking Git status / saving the revised handover, then continue with creation and execution of the baseline script.**
