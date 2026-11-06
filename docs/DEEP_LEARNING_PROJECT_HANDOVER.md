# Food Health Prediction Project --- Deep Learning Handover

*Last updated: 7 November 2026*

This document is the working handover for the Deep Learning (DL) part of
the Food Health Prediction project. Keep it updated as work progresses.
It is intended to let a new ChatGPT conversation or another team member
understand the project, what has been completed, current decisions, and
the next tasks.

> **Important:** This document records the known state from the project
> conversations. Where the latest GitHub or local-machine state has not
> been confirmed, it is explicitly marked as needing verification.

------------------------------------------------------------------------

## 1. Project overview

The project combines three broad areas:

1.  **Big Data Analytics (BDA):** process and integrate large
    food-product datasets, including Open Food Facts (OFF) and USDA
    data, using Apache Spark.
2.  **Deep Learning / Machine Learning:** predict a product's NOVA
    food-processing group (1--4) from its ingredient text.
3.  **Mobile application:** an Android-first app is being considered,
    with Flutter discussed as a possible technology.

The DL work is currently focused on preparing and investigating the
ingredient-text dataset. Model training has **not started yet**.

### What the prediction target means

The target is the product's **NOVA group (1, 2, 3, or 4)**. NOVA
describes the degree/nature of food processing. It should not be
described as a simple healthy-versus-unhealthy classification.

The current modelling question is:

> Can a model predict a product's NOVA group using its ingredient text?

An important project limitation is that ingredient text may not contain
all the information needed to determine processing level. Manufacturing
methods, product context, and label quality can affect the NOVA label.

------------------------------------------------------------------------

## 2. Team and workstream context

The available shared project context indicates:

-   **BDA teammate/workstream:** Apache Spark processing and integration
    of Open Food Facts and USDA data.
-   **DL workstream (this document's owner):** ingredient-text data
    preparation, data-quality investigation, and later ML/DL modelling.
-   **App workstream:** Android-first application; Flutter has been
    discussed.

Historical BDA integration results reported in project conversations:

  Metric                                    Reported value
  --------------------------------------- ----------------
  Matched products                                 368,055
  Products with OFF ingredients                    314,306
  Products with USDA ingredients                   365,741
  Products with both ingredient sources            313,501
  Extra OFF records                                      1

Historical output paths mentioned by the BDA workstream:

-   `data/integration/combined_products.parquet`
-   `data/integration/usda_latest.parquet`

These files may exist only in a teammate's local environment or a
particular branch. Verify their availability before relying on them. The
current DL sample described below was created from `data/food.parquet`,
not confirmed to have come from the integrated BDA output.

The assistant does not automatically see every message or file in
teammates' separate chats. This handover contains only the shared
context available so far. Update it when teammates confirm new decisions
or outputs.

------------------------------------------------------------------------

## 3. Repository and local development setup

### Repository structure observed

``` text
food-facts-project/
├── src/
│   ├── bda/
│   └── deep_learning/
│       ├── prepare_dataset.py
│       ├── check_dataset.py
│       └── investigate_duplicates.py
├── docs/
├── notebooks/
├── notes/
├── data/
├── .venv/
├── .gitignore
└── README.md
```

The `split_dataset.py` file is the next planned file; it had **not yet
been created** at the time this handover was written.

### Local machine and Python environment

Known local setup:

-   Machine: MacBook Air with Apple M1 chip.
-   RAM: 8 GB.
-   Python: 3.12.5, ARM64.
-   Existing project virtual environment: `.venv`.
-   macOS: 26.3 was reported earlier.
-   Keep using the existing `.venv`; do not recreate it without a
    reason.
-   The terminal prompt has shown both `(.venv)` and `(base)`. The
    project `.venv` should be active when running project scripts.

Libraries previously checked:

  Library              Reported version/status
  -------------------- -------------------------
  NumPy                2.5.4
  Pandas               3.1.1
  scikit-learn         1.9.1
  PyArrow              26.0.1
  PyTorch              2.15.0
  PyTorch MPS build    Available
  PyTorch MPS device   Available

The PyTorch device check returned `mps`, meaning Apple GPU acceleration
is available. Actual training performance still needs to be tested with
the chosen model and batch sizes.

### Important resource constraint

The Mac has 8 GB RAM. Do not load the entire multi-million-row Parquet
dataset into Pandas. Use PyArrow batch iteration for large-data scans
and use the small sampled CSV for initial experiments.

------------------------------------------------------------------------

## 4. Source data and known characteristics

The local source file used for the current DL preparation was:

``` text
data/food.parquet
```

Reported characteristics of the current local file:

-   Approximately 4.9 million rows (one scan reported 4,895,629).
-   Relevant fields include `code`, `ingredients`, `nova_group`, and
    `lang`.
-   `ingredients` is structured JSON, not a plain ingredient string.
-   Ingredient objects contain a `text` field and may contain nested
    `ingredients`.
-   `nova_group` contains values 1--4 and missing values.
-   `lang` is a product language field; it is not a guarantee that every
    ingredient phrase is in that language.

A previous language-count scan reported:

  Language           Approximate row count
  ---------------- -----------------------
  English (`en`)                 2,054,773
  French (`fr`)                  1,386,936
  Spanish (`es`)                   385,414
  German (`de`)                    360,625
  Italian (`it`)                   256,815
  Dutch (`nl`)                     103,694

There are other languages too. These are historical scan results and may
vary with dataset versions.

### Why the current sample is English-filtered

The current 10,000-row training sample was generated using
`lang == "en"` as a first baseline. English was chosen to make the first
text-processing/model pipeline simpler to establish.

This is a practical first step, **not a decision to make the final
project English-only**. The `lang` field is only an imperfect proxy:
some sampled ingredient text still contains non-English terms.

Later multilingual work can be considered using multilingual models such
as mBERT or XLM-R, with evaluation broken down by language. Do not claim
multilingual performance until it has actually been tested.

------------------------------------------------------------------------

## 5. Completed DL work

### 5.1 `prepare_dataset.py`

Path:

``` text
src/deep_learning/prepare_dataset.py
```

Purpose:

-   Read the large Parquet file in batches using PyArrow.
-   Avoid loading the full 4.9-million-row dataset into Pandas.
-   Keep English-language product records for the initial baseline.
-   Skip missing ingredient data and invalid/missing NOVA labels.
-   Extract ingredient text recursively from the structured JSON,
    including nested ingredients.
-   Use reservoir sampling to select up to 2,500 examples per NOVA
    class.
-   Write a balanced sample to CSV.

Key implementation details reported:

-   `ParquetFile.iter_batches(...)`
-   Batch size: 5,000 rows.
-   Selected columns: `code`, `ingredients`, `nova_group`, `lang`.
-   `use_threads=False`.
-   Random seed: 42.
-   Output: `data/deep_learning/training_sample.csv`.
-   Output columns: `code`, `ingredient_text`, `nova_group`.

The script scans the full Parquet file but keeps only a small sample in
memory. Reservoir sampling is used so that it does not need to retain
every eligible row.

#### Reported English-filtered counts

  NOVA group     Eligible English records reported
  ------------ -----------------------------------
  1                                         53,594
  2                                          8,684
  3                                         91,175
  4                                        387,061

The script sampled 2,500 from each group, for 10,000 total records.

A previous run without the English filter reported different counts:

  NOVA group     Eligible records reported
  ------------ ---------------------------
  1                                133,848
  2                                 20,672
  3                                217,683
  4                                755,503

The English-filtered run overwrote `training_sample.csv`. The earlier
non-English-filtered sample was not confirmed as separately saved.

### 5.2 `check_dataset.py`

Path:

``` text
src/deep_learning/check_dataset.py
```

Purpose: inspect the sampled dataset for missing values, duplicates,
text lengths, and class balance.

Reported results for `data/deep_learning/training_sample.csv`:

  Check                                                                  Result
  --------------------------------------------------------- -------------------
  Rows                                                                   10,000
  Columns                                                                     3
  Missing `code`                                                              0
  Missing `ingredient_text`                                                   0
  Missing `nova_group`                                                        0
  Duplicate barcodes                                                          0
  Duplicate ingredient descriptions (initial exact check)                 2,315
  Unique descriptions                                                     7,685
  Descriptions shorter than 5 characters                                    143
  Mean ingredient-text length                                 129.55 characters
  Median ingredient-text length                                   56 characters
  Maximum ingredient-text length                               2,551 characters

NOVA distribution:

  NOVA group     Sample count
  ------------ --------------
  1                     2,500
  2                     2,500
  3                     2,500
  4                     2,500

Short descriptions should not automatically be removed. Examples such as
`egg`, `salt`, `oil`, `honey`, or `broccoli` can be valid ingredient
descriptions for single-ingredient products.

A product with the same ingredient description as another product is not
necessarily the same product. Different barcodes can represent different
products.

### 5.3 `investigate_duplicates.py`

Path:

``` text
src/deep_learning/investigate_duplicates.py
```

Purpose: normalize ingredient descriptions and investigate repeated
descriptions that have different NOVA labels.

The initial version printed too many examples because its loop condition
did not stop after a small number of examples. It was replaced with a
concise version that prints a summary and the first 10 conflicting
descriptions.

Latest reported output:

``` text
========== DUPLICATE SUMMARY ==========
Total products: 10000
Products with repeated descriptions: 3382
Repeated unique descriptions: 551
Descriptions with conflicting NOVA labels: 73

========== FIRST 10 CONFLICTING EXAMPLES ==========

Ingredients: _peanuts_, sea salt
NOVA labels: [2, 3]
Number of products: 3

Ingredients: agave nectar
NOVA labels: [1, 2]
Number of products: 5

Ingredients: almonds
NOVA labels: [1, 2]
Number of products: 38

Ingredients: apple cider vinegar, diluted with water to 5% acidity
NOVA labels: [1, 2]
Number of products: 8

Ingredients: apple cider vinegar, filtered water
NOVA labels: [1, 2]
Number of products: 3

Ingredients: apple cider vinegar, water
NOVA labels: [1, 2]
Number of products: 6

Ingredients: arrowroot starch
NOVA labels: [2, 3]
Number of products: 2

Ingredients: avocado
NOVA labels: [1, 2]
Number of products: 4

Ingredients: avocado oil
NOVA labels: [2, 3]
Number of products: 22

Ingredients: balsamic vinegar
NOVA labels: [1, 2]
Number of products: 3
```

Interpretation:

-   3,382 product rows belong to repeated ingredient descriptions.
-   These represent 551 distinct normalized repeated descriptions.
-   73 repeated descriptions have more than one NOVA label.

Possible reasons include product/manufacturing context not captured by
the ingredient text, label inconsistency, or data-quality issues. These
conflicts are not proof that a particular label is wrong.

**Decision:** Do not automatically delete all repeated descriptions or
conflicting-label records. Preserve the data, document the limitation,
and prevent identical ingredient descriptions from leaking across
train/validation/test splits.

The script also previously saved a file named:

``` text
data/deep_learning/duplicate_analysis.csv
```

Verify whether the current version of the script still writes this file
before relying on it.

------------------------------------------------------------------------

## 6. Current data files

Expected local data layout:

``` text
data/
├── food.parquet
└── deep_learning/
    ├── training_sample.csv
    └── duplicate_analysis.csv   # may be generated by the earlier script version
```

The data directory is intentionally excluded from GitHub. These files
are local/generated data and should not be committed unless the team
later makes a deliberate decision to version a small, permitted sample.

Do not assume these local files are available to another team member or
a fresh clone. The preparation script and documentation should make it
possible to regenerate the sample, provided the source data is obtained
separately.

------------------------------------------------------------------------

## 7. Git and GitHub status

Known branch:

``` text
feature/deep-learning
```

The shared team branch is `main`. The DL work should be developed and
pushed to `feature/deep-learning`, not directly to `main`.

A `git status` output showed:

``` text
On branch feature/deep-learning
Your branch is ahead of 'origin/feature/deep-learning' by 4 commits.

Untracked files:
    src/deep_learning/
```

The four commits visible in a later `git log` command were:

``` text
cc0c793 Merge pull request #1 from Ayushbms05/feature/bda
3197165 Corrected the file structure
3139b90 Corrected the file structure
e92ee82 Document BDA integration results
```

The later log showed `cc0c793` at `HEAD -> feature/deep-learning` and
also at `origin/main`. At that time, `origin/feature/deep-learning` was
not shown in the short log output. This needs a fresh check before
assuming the remote branch is current.

The four intended files were staged:

``` text
modified: .gitignore
new file: src/deep_learning/check_dataset.py
new file: src/deep_learning/investigate_duplicates.py
new file: src/deep_learning/prepare_dataset.py
```

The assistant then advised running:

``` bash
git commit -m "Add deep learning dataset preparation and analysis scripts"
git push origin feature/deep-learning
```

**The user has not yet confirmed that these commands were executed
successfully.** Therefore, the first DL commit/push status is unknown
and must be checked.

### Git checkpoint instructions

Before committing:

``` bash
git status
```

Stage specific code/documentation files, not everything blindly. Avoid
`git add .` unless the staged changes have been reviewed.

After a meaningful completed milestone:

``` bash
git add <specific-files>
git commit -m "Clear description of the change"
git push origin feature/deep-learning
```

### `.gitignore`

The intended `.gitignore` content is:

``` gitignore
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

The user showed an earlier version where `.DS_Store` appeared with a
trailing `%` in the pasted terminal output. This may have been a
prompt/paste artifact; verify the actual file in VS Code. The user was
advised to add `*.csv` and `*.parquet` to prevent generated datasets
from being committed.

Note: broad `*.csv` and `*.parquet` patterns exclude all such files
anywhere in the repository. If the team later needs to version a small
example data file, review the ignore rules and use a deliberate
exception.

### Syncing with `main`

Recommended workflow when the working tree is clean and local work is
committed/pushed:

``` bash
git switch feature/deep-learning
git fetch origin
git merge origin/main
git push origin feature/deep-learning
```

Sync at sensible milestones: before a major task, after teammates merge
significant changes into `main`, and before opening a pull request.
Merge `main` into the feature branch; do not push DL changes directly to
`main`.

------------------------------------------------------------------------

## 8. Planned modelling approach

The planned model progression is:

1.  **Baseline:** TF-IDF features + Logistic Regression (traditional ML
    baseline).
2.  **Deep Learning:** LSTM implemented in PyTorch.
3.  **Possible later experiment:** a compact transformer, such as
    DistilBERT; for multilingual experiments consider mBERT or XLM-R.

Do not jump straight to a complex transformer. Establish a reproducible
baseline first.

### Text representation

Ingredient text cannot be fed directly to a conventional neural network
as ordinary strings. It must be transformed into numeric input:

-   TF-IDF converts text into numerical feature vectors.
-   LSTM models use token IDs and embeddings.
-   Transformer models use their own tokenizer and token IDs/attention
    masks.

### Evaluation

Report more than accuracy. Planned metrics:

-   Accuracy
-   Precision
-   Recall
-   Macro-F1
-   Confusion matrix
-   Per-class results

Because the 10,000-row sample is balanced by design but the underlying
English-labelled population is strongly imbalanced, the balanced sample
is useful for initial experiments but does not represent the natural
class distribution. Later evaluation should include a
realistic/natural-distribution test set or clearly report the limitation
and use additional representative evaluation.

------------------------------------------------------------------------

## 9. Immediate next task: create train/validation/test splits

**Status: planned, not yet implemented.**

The next file to create:

``` text
src/deep_learning/split_dataset.py
```

Input:

``` text
data/deep_learning/training_sample.csv
```

Expected output files (suggested):

``` text
data/deep_learning/
├── training_sample.csv
├── train.csv
├── validation.csv
└── test.csv
```

Initial target proportions:

-   Train: approximately 70%
-   Validation: approximately 15%
-   Test: approximately 15%

The exact row counts may differ because groups of identical normalized
ingredient descriptions must remain together.

### Critical splitting requirement

Create a normalized ingredient-text grouping key, for example:

-   lowercase text
-   trim leading/trailing whitespace
-   collapse repeated whitespace

All rows with the same normalized ingredient description must be
assigned to the same partition. This prevents the model from seeing the
same ingredient text during training and then being tested on that exact
text.

Do not simply use a random row-level split.

Potential scikit-learn tools include `GroupShuffleSplit` or
`StratifiedGroupKFold`. Choose and validate the approach carefully
because some ingredient-text groups contain conflicting NOVA labels.
Check the class distribution in each split and ensure all four classes
are represented. If using sequential group splits, use fixed random
seeds for reproducibility and inspect the resulting counts.

The split script should save the split files under
`data/deep_learning/`, print row counts and NOVA distributions, and
verify that no normalized ingredient group appears in more than one
split.

Do not begin model training until the split checks pass.

------------------------------------------------------------------------

## 10. Suggested next milestones and Git checkpoints

  -----------------------------------------------------------------------------
  Milestone               Work                          Git action
  ----------------------- ----------------------------- -----------------------
  Dataset preparation     `prepare_dataset.py` works    Commit/push
                          and output is reproducible    

  Dataset quality checks  `check_dataset.py` completed  Commit/push

  Duplicate investigation `investigate_duplicates.py`   Commit/push
                          completed                     

  Handover documentation  Keep this document updated    Commit/push

  Dataset splitting       `split_dataset.py` and        Commit/push
                          leakage checks completed      

  Baseline model          TF-IDF + Logistic Regression  Commit/push

  LSTM                    PyTorch training and          Commit/push
                          validation                    

  Evaluation              Metrics, confusion matrix,    Commit/push
                          findings                      
  -----------------------------------------------------------------------------

The current preparation/inspection scripts and this document are
appropriate GitHub content. Do not commit `.venv/`, the large source
Parquet file, generated CSV datasets, or other large local data outputs.

------------------------------------------------------------------------

## 11. Working style and instructions for the next assistant conversation

The user is a beginner in deep learning and prefers patient, incremental
guidance.

Please:

1.  Give one manageable task at a time. Do not dump the entire project
    plan as a list of commands to execute at once.
2.  Clearly label where each action happens:
    -   **VS Code → Editor** for creating/editing code.
    -   **VS Code → Integrated Terminal** for commands.
3.  Explain briefly what a command or code section does and why it is
    needed.
4.  Wait for the user's output before proceeding to the next step when
    the task depends on that output.
5.  Keep track of Git milestones. Explicitly tell the user when to
    commit and push, and which files to include.
6.  Do not tell the user to stage everything blindly. Check `git status`
    and stage specific files.
7.  Keep datasets and virtual environments out of Git.
8.  Do not assume the user has successfully committed or pushed unless
    they show the successful output.
9.  Explain technical terms simply, without sacrificing correctness.
10. When integrating with BDA, verify the actual dataset schema and
    provenance instead of assuming that historical output files are
    present.

### Current exact stopping point

The user agreed to proceed with the train/validation/test split, but
then paused and asked for this handover document. The assistant had
started to introduce the split proportions (70/15/15) but **had not yet
provided the split code**. No `split_dataset.py` creation or execution
has been confirmed.

The immediate next action after saving this document to GitHub is to
verify Git status and commit/push the document and any still-uncommitted
DL scripts. Then continue by creating and testing `split_dataset.py` in
small steps.
