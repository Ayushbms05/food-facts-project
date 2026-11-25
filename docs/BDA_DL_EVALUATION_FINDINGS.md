# Food Facts — BDA + DL Integration and Evaluation Findings

**Living technical report | Revision 0.1 | 25 November 2026 | Completed through I6.2**

Repository: `Ayushbms05/food-facts-project`  
Working branch: `feature/bda-dl-integration`  
Scope: integration of Member 1's BDA pipeline and Member 2's trained NOVA classifier. Member 3's Kotlin/XML work remains separate at this stage.

## Executive summary

The BDA and DL source branches have been merged into a separate integration branch and pushed to GitHub. The existing DistilBERT checkpoint loads and predicts locally on Apple MPS. A reproduced 10-product smoke test returned 10/10 correct predictions; this is a functional test, not a new generalization accuracy measurement.

Two sequential audits screened the BDA-curated OFF–USDA intersection and the India-tagged OFF dataset for previously used product identifiers and ingredient descriptions. The second audit also recovered the original structured ingredient representation used for DL training and checked source-record consistency and English metadata. The final pools contain **252,365 curated products**, **5,782 India-tagged products**, and **252,101 eligible OFF–USDA pairs** (a subset of the curated pool). No model predictions on these new pools or fine-tuning have been performed yet.

The main finding is that screening by raw and cleaned ingredient text alone was insufficient: the structured audit found 28,236 further curated candidates whose structured text matched an earlier DL example. The resulting class imbalance, particularly the 909 curated and 67 India NOVA 2 candidates, must guide evaluation design.

## 1. Research objectives and reference baseline

1. Establish whether the transferred model works on the BDA machine.
2. Prepare an evaluation set without known exact identifier or ingredient-text overlap with earlier DL experiments.
3. Compare raw OFF text, BDA-cleaned OFF text and training-style structured OFF text on the same products.
4. Measure prediction consistency across matched OFF and USDA descriptions; do not treat the OFF NOVA label as independently validated USDA ground truth.
5. Evaluate India-tagged products and, if feasible, assess sensitivity to OCR errors.
6. Use the resulting evidence to decide whether model improvement is necessary before backend integration.

NOVA 1–4 refers to degree and purpose of food processing, not a standalone nutrition or health rating.

### 1.1 Member 2's prior model results (reference, not newly reproduced)

Source: `docs/MODEL_EXPERIMENT_RESULTS.md`, sections 6, 7 and 12; `docs/DEEP_LEARNING_PROJECT_HANDOVER.md`.

| Model | Balanced test accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 82.01% | 0.8188 |
| 1D CNN | 88.94% | 0.8889 |
| Bidirectional LSTM | 89.81% | 0.8977 |
| DistilBERT | 91.34% | 0.9132 |

All four results refer to the same 1,501-record test partition. DistilBERT was selected for integration. On Member 2's separate 10,000-record natural-distribution sample, it achieved 90.11% accuracy and 0.6933 macro F1. NOVA 2 precision was 0.13, recall 0.83 and F1 0.22 (35 actual NOVA 2 products; 29 of 226 NOVA 2 predictions correct). This motivates class-wise reporting rather than reliance on accuracy alone.

## 2. I1–I3 — Artifact transfer, source integration and environment setup

### I1: Artifact transfer

- Received `FoodFacts_DL_Artifacts.zip` (245M as displayed by macOS `ls -lh`). ZIP integrity test passed; extraction used `unzip -n` to avoid overwrites.
- Extracted directory `data/deep_learning/` occupied 278M (`du -sh`); checkpoint weights displayed 255M (`ls -lh`). These shell sizes are distinct from the benchmark's reported model-directory size of 256.11 MB.
- Five dataset CSVs, CNN and LSTM checkpoints/vocabularies, DistilBERT checkpoint/tokenizer/configuration and evaluation reports were present.
- BDA's original OFF file was retained. BDA and DL previously reported different raw-file hashes and row counts (4,893,632 vs 4,895,629); these establish different reported files, not which snapshot is newer or the exact row-level difference.

### I2: Git integration

| Item | Recorded result |
|---|---|
| BDA source tip | `7d6c518` |
| DL source tip | `5b8614f` |
| Integration branch | `feature/bda-dl-integration` |
| Merge | No conflicts; 30 files changed, 7,470 insertions, 1 deletion |
| Remote tracking | `origin/feature/bda-dl-integration` confirmed |
| Git status after push | Clean |

Updated root `.gitignore` to exclude the integration environment and Android build/cache files. No raw data, trained model files or mobile build outputs intentionally staged for integration. Original BDA/DL/mobile branches were not merged into `main`. The current integration commit hash has not been captured in this report.

### I3: Environment verification

The existing BDA `.venv` was retained. Created `.venv-integration` using Homebrew Python 3.12.16 on arm64. Dependency dry run succeeded and imports confirmed all eight pinned versions:

| Library | Version |
|---|---|
| PyTorch | 2.15.0 |
| Transformers | 5.24.0 |
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| Scikit-learn | 1.9.1 |
| PyArrow | 26.0.1 |
| Matplotlib | 3.11.2 |
| Safetensors | 0.8.0 |

MPS built and available: true; CUDA available: false; selected device: MPS. Tensor multiplication test returned `[2.0, 4.0, 6.0]`. Import and computation tests passed; a standalone `pip check` result was not included in the supplied logs.

## 3. I4 — Saved-model inference and performance

Commands (repository root, `.venv-integration` active):

```bash
HF_HUB_OFFLINE=1 python -m src.deep_learning.inference.predict
HF_HUB_OFFLINE=1 python -m src.deep_learning.evaluation.test_predictor
HF_HUB_OFFLINE=1 python -m src.deep_learning.evaluation.benchmark_model_performance
```

### 3.1 Interactive smoke test

| Ingredient input | Predicted NOVA | Top Softmax score |
|---|---:|---:|
| Whole almonds | 1 | 79.25% |
| Refined sunflower oil | 2 | 99.12% |
| Chickpeas, water, salt | 3 | 98.85% |
| Refined wheat flour, sugar, palm oil, emulsifier E471, artificial flavour, glucose syrup | 4 | 98.85% |

All four matched illustrative expectations. The saved-model test also reproduced 10/10 correct predictions on the existing test examples. Neither test estimates accuracy on a new population. Softmax scores are not independently calibrated probabilities of correctness.

The macOS multiprocessing redirects warning did not prevent execution. A `KeyboardInterrupt` occurred when the interactive script was exited with Ctrl+C; the script also supports `exit`.

### 3.2 Cross-machine benchmark

| Metric | Member 2 (M1, 8 GB) | Member 1 (Apple arm64) |
|---|---:|---:|
| Model-directory size (reported MB) | 256.11 | 256.11 |
| Loading time (s) | 2.48 | 0.82 |
| Average inference (ms) | 57.26 | 13.10 |
| Median inference (ms) | 51.20 | 12.97 |
| Minimum inference (ms) | 32.75 | 12.38 |
| Maximum inference (ms) | 205.83 | 15.10 |
| Approximate predictions/s | 17.46 | 76.34 |
| Device | MPS | MPS |
| Measured predictions | 50 | 50 |

Member 1's run achieved about 4.37 times the reported prediction throughput. These are individual local benchmark runs, not controlled hardware experiments or end-to-end API latency measurements. Hardware, caches and background load may influence timings. Both reports were preserved and verified at `data/deep_learning/reports/model_performance_benchmark_member1.txt` and `model_performance_benchmark_member2.txt` (same directory).

## 4. I5 — Dataset compatibility inspection

| Dataset | Path | Columns | Role |
|---|---|---:|---|
| Raw OFF | `data/food.parquet` | 111 | Original records, including structured ingredients and language metadata |
| Integrated OFF–USDA | `data/integration/combined_products.parquet` | 11 | Barcode-matched source descriptions |
| Curated BDA | `data/curated/source_aware_products.parquet` | 18 | Raw/clean text, source agreement, label and candidate flags |
| India candidates | `data/india_candidates.parquet` | 5 | India-tagged OFF records with ingredient text and NOVA information |

The raw OFF `ingredients_text` field is a list of language/text structs, while `ingredients` is a JSON string. DL's `extract_ingredients()` parses the JSON and recursively collects parent and nested ingredient `text` values, joining them with commas. Training required product-level `lang == 'en'`. BDA extracts English original label text and separately comparison-cleans it. The BDA candidate flag requires nonempty `off_clean` and an OFF NOVA label from 1 to 4; it is not proof of unseen data, language correctness or label quality.

### DL CSV inspection

| Dataset | Rows | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
|---|---:|---:|---:|---:|---:|
| training_sample.csv | 10,000 | 2,500 | 2,500 | 2,500 | 2,500 |
| train.csv | 7,018 | 1,742 | 1,758 | 1,767 | 1,751 |
| validation.csv | 1,481 | 375 | 369 | 368 | 369 |
| test.csv | 1,501 | 383 | 373 | 365 | 380 |
| natural_distribution_sample.csv | 10,000 | 562 | 35 | 1,615 | 7,788 |

All five files had columns `code`, `ingredient_text`, `nova_group`; no missing codes or empty ingredient descriptions; unique raw codes within each file. The balanced sample is the parent of the three partitions and is not counted as another 10,000 independent examples.

## 5. I6.1 — Preliminary exact overlap audit

**Question:** Would evaluation on BDA data repeat products or ingredient descriptions already used in DL experiments?

**Method:** `src/integration/audit_bda_dl_overlap.py` reads train, validation, test and natural-distribution CSVs. Product codes remain strings; numeric codes up to 14 digits are left-padded to 14 for conservative exclusions (not GTIN checksum validation). Text keys are lowercased and whitespace-collapsed. The script excludes missing identifiers and exact matches on normalized codes or raw/clean OFF text. USDA text matches are counted separately and affect paired eligibility, not the OFF-only pool. Parquet is read in batches of 4,096 rows.

### 5.1 Previous partition overlap

20,000 rows; 20,000 unique normalized codes; 17,047 unique normalized ingredient descriptions. All six pairwise comparisons among train, validation, test and natural sample returned zero intersecting normalized codes and zero intersecting normalized texts. Within-file repeated descriptions can still exist. This does not exclude near duplicates or pretraining exposure.

### 5.2 Preliminary filtering results

| Metric | Curated | India |
|---|---:|---:|
| Rows inspected | 368,055 | 7,101 |
| Eligible | 302,686 | 7,048 |
| Missing normalized code | 0 | 0 |
| Barcode overlap | 11,329 | 292 |
| Raw OFF text overlap | 3,191 | 681 |
| Clean OFF text overlap | 9,666 | Not applicable |
| USDA text overlap | 10,327 | Not applicable |
| Unique rows excluded from OFF evaluation | 20,794 | 890 |
| Remaining OFF candidates | 281,892 | 6,158 |
| Remaining OFF–USDA paired candidates | 281,136 | Not applicable |

Overlap reason counts can intersect; the union is the actual number of OFF exclusions. Of retained curated candidates, 64,402 were flagged for review. Agreement categories: `same_basic` 151,279; `same_after_extra_cleanup` 66,211; `still_different` 64,234; `off_only` 168.

**Finding:** Existing DL examples were present within the BDA datasets, so direct evaluation without screening could inflate apparent generalization. This audit did not check the training-style structured text.

Evidence: `data/integration/audits/i6_1_preliminary_overlap.json`.

## 6. I6.2 — Structured ingredient compatibility and overlap audit

**Question:** Do preliminary candidates still match previous DL descriptions when expressed in the exact training format, and can we reliably recover that format from the original OFF record?

**Method:** `src/integration/prepare_bda_dl_evaluation.py` reconstructs the I6.1 preliminary pools and checks their counts against the prior JSON. It joins candidate `off_code`/`code` to original OFF codes (exact after surrounding whitespace trim, not a padded-code join). It reads 4,096-record batches, stores matches in temporary SQLite, and reuses the DL training `extract_ingredients()` function. It checks unique normalized candidate identifiers, unique original record, matching NOVA label, `lang == 'en'`, matching raw English text, prior raw text overlap, nonempty structured text and prior structured text overlap. Retained rows preserve all available representations and source metadata. Exclusions are mutually exclusive first-failure reasons in script order, not counts of every possible problem.

### 6.1 Raw source scan

| Measurement | Result |
|---|---:|
| Original OFF rows scanned | 4,893,632 |
| Unique exact codes requested | 288,033 |
| Exact codes matched | 288,033 |
| Ambiguous exact codes | 0 |
| Retained normalized-code overlap across cohorts | 15 |

All requested codes matched. No first-failure label mismatch, raw text mismatch or duplicate candidate identifier was reported. The two pools share 15 normalized codes, so they are not independent populations.

### 6.2 Decision counts

| Decision | Curated | India |
|---|---:|---:|
| Preliminary candidates | 281,892 | 6,158 |
| Structured text overlap | 28,236 | 252 |
| Original language not English | 1,146 | 116 |
| Missing structured text | 40 | 8 |
| Additional original raw text overlap | 105 | 0 reported |
| Total excluded in I6.2 | 29,527 | 376 |
| Retained | 252,365 | 5,782 |

Structured-text overlap accounts for 95.63% of curated I6.2 exclusions. I6.2 retains 89.53% of the curated preliminary pool and 93.89% of the India preliminary pool. Across both audits, curated exclusions total 50,321 (83.38% retained, rounded); India exclusions total 1,266 (82.04% retained).

### 6.3 Class distributions through both audits

| NOVA | Curated eligible | After I6.1 | After I6.2 | India eligible | After I6.1 | After I6.2 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 29,375 | 20,680 | 11,418 | 1,288 | 718 | 550 |
| 2 | 5,053 | 2,252 | 909 | 262 | 111 | 67 |
| 3 | 57,034 | 53,508 | 44,222 | 1,105 | 1,042 | 991 |
| 4 | 211,224 | 205,452 | 195,816 | 4,393 | 4,287 | 4,174 |
| **Total** | **302,686** | **281,892** | **252,365** | **7,048** | **6,158** | **5,782** |

The curated pool is 77.59% NOVA 4. Only 17.99% of initial curated NOVA 2 candidates survive both audits. This is a selection effect, not evidence of poor classification. India has only 67 NOVA 2 candidates, so class estimates may have substantial uncertainty. The initial 1,000-per-class proposal is no longer feasible.

### 6.4 Paired-source candidate pool

| Group | Eligible pairs |
|---|---:|
| NOVA 1 (OFF label) | 11,271 |
| NOVA 2 (OFF label) | 901 |
| NOVA 3 (OFF label) | 44,166 |
| NOVA 4 (OFF label) | 195,763 |
| **Total** | **252,101** |

| Source agreement category | Eligible pairs |
|---|---:|
| same_basic | 130,103 |
| same_after_extra_cleanup | 61,784 |
| still_different | 60,214 |
| **Total** | **252,101** |

264 retained curated rows are not pair-eligible. Pair eligibility checks nonempty USDA raw text and no exact prior DL match in USDA raw/clean text. It does not verify language, formulation equivalence or USDA labels. Even a shared barcode can represent different ingredient formulations or versions. `same_basic` and `same_after_extra_cleanup` mean string agreement under BDA cleaning, not established semantic equivalence.

### 6.5 Generated evidence files

Directory: `data/integration/evaluation_i6_2/`.

| File | Observed size (`ls -lh`) | Role |
|---|---|---|
| curated_candidates.csv | 390M | 252,365 screened curated candidates |
| india_candidates.csv | 3.2M | 5,782 screened India candidates |
| i6_2_structured_audit.json | 2.2K | Audit decisions and counts |

Candidate CSV columns: `cohort`, `code`, `normalized_code`, `nova_group`, `product_name`, `off_raw`, `off_clean`, `off_dl_structured`, `usda_raw`, `usda_clean`, `agreement_category`, `needs_review`, `off_records_for_gtin`, `pair_eligible`. India rows do not have a BDA-cleaned representation, USDA pair or populated product name in this implementation. Original data is read-only; generated files are stored separately. Temporary SQLite is removed by the script.

**Finding:** Recovering training-style structured text materially changed the candidate pool. The audits are complete for this stage, but final sampling and evaluation remain pending. No inference or training was performed by I6.1 or I6.2.

## 7. Limitations and interpretation rules

- Exact normalized text and identifier exclusion reduces known overlap; it cannot establish complete absence of near duplicates, alternate formulations or pretrained-model exposure.
- Product-level English metadata does not guarantee English ingredient text; the pool is not a verified multilingual or English-only gold standard.
- Both BDA and DL draw from OFF; NOVA values are recorded OFF labels, not independently adjudicated truth. This is a screened same-source evaluation rather than independent external validation.
- Existing barcode normalization is conservative and is not itself checksum validation. Short OFF identifiers may not be globally valid GTINs.
- Filtering disproportionately reduces some NOVA classes and changes the distribution. Natural sampling from the retained pool is representative of that pool only, not all retail foods.
- Same normalized text may remain repeated within a candidate pool or across new cohorts. Check duplicates and conflicting labels before final sampling; report exact rules and remaining overlap. Do not aggregate overlapping cohorts as independent samples.
- India tags do not establish country of manufacture or represent all Indian retail products. Only the selected OFF data and language scope are being evaluated.
- USDA pairs have no independent NOVA reference in this workflow; report prediction agreement, not USDA accuracy against an assumed transferred OFF label.
- Benchmark latency is not backend latency; model Softmax scores are not guarantees of correctness.
- Report values are transcribed from supplied terminal runs and Member 2's documentation. The full datasets and model were not independently rerun for this report.

## 8. Next experiments — proposed, not completed

| Step | Planned experiment | Status |
|---|---|---|
| I6.3a | Audit duplicate ingredient keys and conflicting labels within/between retained pools; freeze reproducible sample manifests | Next |
| I6.3b | Balanced curated evaluation, same products across raw, cleaned and structured OFF inputs | Pending |
| I6.3c | Natural-distribution curated evaluation from screened population | Pending |
| I6.4 | OFF–USDA paired prediction consistency, stratified by source agreement | Pending |
| I6.5 | India-tagged evaluation | Pending |
| I6.6 | OCR-to-DL paired comparison where reference transcription exists | Pending |
| I7 | Decide whether model improvement is justified | Pending |
| I8–I11 | Backend, Android integration, real scan-event analytics, final tests/documentation | Pending |

Provisional design: target up to 500 per class for a balanced curated sample (adjust to actual unique-text counts), 10,000 for a natural sample if feasible, and about 2,000 source pairs stratified by agreement category. Evaluate India candidates subject to final overlap and duplicate checks. Fix a seed (proposed 42), preserve barcodes as strings, document selection and any cross-cohort overlap. Balanced and natural samples have distinct purposes; do not infer population precision from a balanced sample. Reserve untouched evaluation records if later tuning is needed. Report accuracy, macro F1, class-wise precision/recall/F1/support, confusion matrices, paired differences and prediction confidence diagnostics as appropriate. No final sample sizes or improvement decisions have been established.

## 9. Reproducibility and report maintenance

Keep this file in `docs/BDA_DL_EVALUATION_FINDINGS.md` on `feature/bda-dl-integration`. Commit the report and the experiment scripts, not the generated datasets or checkpoint weights. Generated evidence under `data/` stays local; retain it or archive it separately for reproducibility. For each new completed experiment append date, objective, script and exact command, input/model identifiers (including hashes where available), sample size and seed, results, output paths, interpretation and limitations. Preserve prior findings; mark corrections explicitly rather than silently replacing results. Use Git history to retain versions. The report is updated when new experiment outputs are supplied, not automatically by a background process.

### Run commands for completed audits

```bash
source .venv-integration/bin/activate
python src/integration/audit_bda_dl_overlap.py
python src/integration/prepare_bda_dl_evaluation.py
```

These commands recreate the generated audit outputs; archive earlier runs if comparing changed inputs or methods.

### Revision history

| Revision | Date | Scope |
|---|---|---|
| 0.1 | 2026-11-25 | Initial report: I1–I5 setup and verification, I6.1 preliminary overlap, I6.2 structured audit, candidate distributions and next experimental design |

**Current conclusion:** Keep the existing trained DistilBERT checkpoint. The integration and screening results justify proceeding to evaluation, but do not yet justify retraining or claims of accuracy on BDA-curated or India-tagged data.
