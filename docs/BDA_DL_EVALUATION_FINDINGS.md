# Food Facts — BDA + DL Integration and Evaluation Findings

**Research closeout | Revision 1.0 | 26 November 2026 | I1–I7 complete; backend documented separately**

Repository: `Ayushbms05/food-facts-project`  
Working branch: `feature/bda-dl-integration`  
Scope: integration of Member 1's BDA pipeline and Member 2's trained NOVA classifier. Member 3's Kotlin/XML work remains separate at this stage.

## Executive summary

**Report boundary:** This report covers the completed BDA–DL integration and evaluation work through I7. Backend development (I8 onward) is now maintained in `docs/BACKEND_DEVELOPMENT_REPORT.md`. The earlier SQLite/PyArrow catalogue proposal was superseded before execution; the approved direction is Spark ETL, PostgreSQL serving, FastAPI, Kafka, Spark Structured Streaming and Streamlit. This is an architecture decision, not a claim of completed implementation.

**Completed research status (I7.4–I7.6):** The frozen combined adjustment passed the four predeclared holdout gates. On 5,080 structured curated products, accuracy rose from 92.1654% to 92.6181% (+23 correct), and NOVA 2 false positives declined from 97 to 67 with two fewer true positives. The original checkpoint remains unchanged. Recommend no immediate fine-tuning; retain the original model for general app inference, with the combined adjustment restricted to an explicitly identified screened-curated structured-text research mode. India and OCR improvement remains unvalidated. The policy file is now present locally; backend implementation remains pending.

The BDA and DL source branches have been merged into a separate integration branch and pushed to GitHub. The existing DistilBERT checkpoint loads and predicts locally on Apple MPS. A reproduced 10-product smoke test returned 10/10 correct predictions; this is a functional test, not a new generalization accuracy measurement.

Two sequential audits screened the BDA-curated OFF–USDA intersection and the India-tagged OFF dataset for previously used product identifiers and ingredient descriptions. The second audit also recovered the original structured ingredient representation used for DL training and checked source-record consistency and English metadata. The final pools contain **252,365 curated products**, **5,782 India-tagged products**, and **252,101 eligible OFF–USDA pairs** (a subset of the curated pool). At I6.2, no model predictions or fine-tuning had been performed.

An audit finding is that screening by raw and cleaned ingredient text alone was insufficient: the structured audit found 28,236 further curated candidates whose structured text matched an earlier DL example. The resulting class imbalance, particularly the 909 curated and 67 India NOVA 2 candidates, must guide evaluation design.

I6.3a subsequently grouped the screened candidates by connected exact normalized OFF text and code, removed 364 conflicting-label groups, and froze three disjoint samples: **2,000 balanced curated**, **10,000 natural curated**, and **5,311 India-tagged** representatives. These are preparation findings, not model-performance results.

I6.3b evaluated all 2,000 balanced curated products in three ingredient representations with the unchanged checkpoint. Structured text achieved **89.60% accuracy / 0.8966 macro F1**, original OFF text **88.40% / 0.8847**, and comparison-cleaned text **60.95% / 0.5567**. The large degradation after comparison cleaning motivates separate BDA matching and DL inference text representations. Structured text is the provisional preferred format when available; raw text remains the practical alternative. I6.3c tested both on the separate frozen natural cohort (10,000 products): structured text achieved **93.01% accuracy / 0.7409 macro F1**, raw text **91.10% / 0.7239**. Structured text is preferred for OFF inference; raw is the fallback. NOVA 2 precision remains weak (25 of 130 structured predictions correct, 19.23%). No training or weight tuning has occurred.

I6.4 evaluated 2,000 matched OFF–USDA pairs (4,000 predictions) drawn from the frozen natural cohort. Raw-input prediction agreement was **95.60%** overall, 100% for `same_basic`, 97.15% for `same_after_extra_cleanup`, and 89.64% for `still_different`. These are stratified consistency results, not USDA accuracy or independent population estimates. The source-aware review flags remain useful; source disagreement does not identify the correct classification.

I6.5 evaluated all 5,311 frozen India-tagged representatives (10,622 calls). Structured text achieved **80.06% accuracy / 0.6255 macro F1**, versus raw text **79.87% / 0.6258**. The format advantage is small and metric-dependent; structured accuracy is 12.95 percentage points below the natural curated cohort, a descriptive comparison across differently selected cohorts. Structured NOVA 2 precision was only **13.77%** (50 correct out of 363 predicted), reinforcing the need for error analysis and independently held-out evaluation of any future adjustments. No training or weight tuning occurred.

I6.6 completed a six-image ingredient-region OCR robustness pilot. PSM 6 yielded 64.21% pooled WER and 3/6 reference-prediction agreements; PSM 11 yielded 71.58% WER and 2/6 agreements including four empty outputs. Meaningless OCR text (`aoa`, `=`) elicited NOVA 1 scores of 90.8% and 80.0%. This motivates input validation and user review, not reliance on confidence alone. Three images overlap the India cohort by barcode, so the pilot is not an independent generalization test. No model weights were changed.

I7.1 audited 30,622 saved predictions across 15,311 products. Structured-text errors with confidence at least 90% numbered 421/699 (natural curated) and 485/1,059 (India). At a descriptive 95% NOVA 2 score cut, precision was still only 54.29% and 32.33%, respectively. These findings motivate fresh development/holdout sets for calibration and prior-adjustment trials, not a threshold fitted to the inspected cohorts. No new inference or tuning was performed.

I7.2b resolved the supplied DL artifact discrepancy at the normalized-record level: all 10,000 code/text/label records agree after normalization, and none of the 17,311 earlier classification evaluation rows overlaps the full training sample by normalized code or any checked OFF text. The exact cause of formatting differences and actual checkpoint lineage remain unverified. I7.3 subsequently evaluated 11 output-adjustment strengths on 5,080 development products. The selected configuration (alpha=0.3, T=1.326715596542784) yielded 93.01% unweighted accuracy versus 92.72% baseline, but NOVA 2 recall fell from 72.50% to 63.75%. Standardized macro F1 increased from 0.7289 to 0.7361; standardized ECE fell from 0.03330 to 0.01080. Temperature-only T=1.2877402988474436 retained all class decisions and improved calibration. These were development results, not independent improvement evidence. The separate holdout was subsequently evaluated in I7.4.

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
- I6.3a resolved known exact normalized OFF text/code group overlap across the frozen samples, as described in Section 10. Near duplicates and USDA text overlap remain possible. Do not aggregate cohorts as independently representative retail samples.
- India tags do not establish country of manufacture or represent all Indian retail products. Only the selected OFF data and language scope are being evaluated.
- USDA pairs have no independent NOVA reference in this workflow; report prediction agreement, not USDA accuracy against an assumed transferred OFF label.
- Benchmark latency is not backend latency; model Softmax scores are not guarantees of correctness.
- Report values are transcribed from supplied terminal runs and Member 2's documentation. The full datasets and model were not independently rerun for this report.

## 8. Experiment status and transition to backend

| Step | Planned experiment | Status |
|---|---|---|
| I6.3a | Audit duplicate ingredient keys and conflicting labels within/between retained pools; freeze reproducible sample manifests | Completed (Section 10) |
| I6.3b | Balanced curated evaluation, same products across raw, cleaned and structured OFF inputs | Completed (Section 11) |
| I6.3c | Natural-distribution curated evaluation, raw and structured OFF inputs | Completed |
| I6.4 | OFF–USDA paired prediction consistency, stratified by source agreement | Completed (Section 13) |
| I6.5 | India-tagged evaluation | Completed (Section 14) |
| I6.6 | OCR-to-DL paired comparison where reference transcription exists | Completed: six-image robustness pilot (Section 15) |
| I7 | Diagnose errors, prepare fresh samples, test and validate output adjustments, decide model policy | Complete (Sections 17–26) |
| I8–I11 | Backend, Android integration, real scan-event analytics, final tests/documentation | Transferred to `docs/BACKEND_DEVELOPMENT_REPORT.md` |

The frozen sampling design (Section 10) yielded 500 per class for the balanced curated sample, 10,000 natural curated representatives, and 5,311 reserved India representatives. The OFF–USDA experiment reached its target of 2,000 pairs (Section 13). Balanced and natural samples have distinct purposes; do not infer population precision from a balanced sample. Reserve untouched groups if later tuning is needed. Report accuracy, macro F1, class-wise precision/recall/F1/support, confusion matrices and paired differences. I7 established that no additional weight training is planned for the prototype; the combined output adjustment passed the curated holdout gates only. Comparison-cleaned input remains excluded from classifier input because of I6.3b.

## 9. Reproducibility and report maintenance

Keep this file in `docs/BDA_DL_EVALUATION_FINDINGS.md` on `feature/bda-dl-integration`. Commit the report and the experiment scripts, not the generated datasets or checkpoint weights. Generated evidence under `data/` stays local; retain it or archive it separately for reproducibility. This research report is closed at I7; retain any future research corrections explicitly. Record new backend, streaming and application experiments in the separate backend report with date, objective, commands, versions, inputs, results, output paths and limitations. Preserve prior findings; mark corrections explicitly rather than silently replacing results. Use Git history to retain versions. The report is updated when new experiment outputs are supplied, not automatically by a background process.

### Run commands for completed audits

```bash
source .venv-integration/bin/activate
python src/integration/audit_bda_dl_overlap.py
python src/integration/prepare_bda_dl_evaluation.py
python src/integration/prepare_i6_3_samples.py
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_3_balanced
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_3_natural
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_4_sources
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_5_india
```

These commands recreate the generated audit outputs; archive earlier runs if comparing changed inputs or methods.

### Revision history

| Revision | Date | Scope |
|---|---|---|
| 0.1 | 2026-11-25 | Initial report: I1–I5 setup and verification, I6.1 preliminary overlap, I6.2 structured audit, candidate distributions and next experimental design |
| 0.2 | 2026-11-25 | I6.3a exact group audit, conflict exclusions, frozen samples and hashes; I6.3b plan |
| 0.3 | 2026-11-25 | I6.3b paired balanced evaluation, class metrics, model provenance, interpretation and I6.3c plan |
| 0.4 | 2026-11-25 | I6.3c natural paired evaluation, class imbalance, NOVA 2 limitation and I6.4 plan |
| 0.5 | 2026-11-25 | I6.4 source consistency results, selection evidence, transition matrices and India evaluation plan |
| 0.6 | 2026-11-25 | I6.5 India-tagged paired evaluation and limitations; I6.6 preparation |
| 0.6.1 | 2026-11-25 | I6.6 local OCR inventory and planned ingredient-region protocol; no new inference results |
| 0.6.2 | 2026-11-25 | OCR crop and reference preparation |
| 0.7 | 2026-11-25 | OCR evaluation completed |
| 0.8 | 2026-11-25 | I7.1 error analysis |
| 0.9 | 2026-11-25 | I7.2 sampling and qualitative review |
| 0.10 | 2026-11-25 | I7.2b provenance and I7.3 protocol |
| 0.11 | 2026-11-26 | I7.3 results and predeclared I7.4 holdout decision gate |
| 0.12 | 2026-11-26 | I7.4 holdout results and I7.5–I7.6 inference policy |
| 0.13–0.14 | 2026-11-26 | Historical I8 inventory, source inspection and unexecuted SQLite design; backend material moved to separate report |
| 1.0 | 2026-11-26 | Research closeout through I7; revised backend direction and report separation, no new evaluation results |

**Historical conclusion at I6.5:** Keep the existing trained DistilBERT checkpoint. Balanced and natural evaluations are complete. Structured text leads raw text on both, and comparison-cleaned text substantially underperforms on the balanced cohort. NOVA 2 false positives remain a concern. Cross-source and India evaluation are complete; India performance is lower and the format difference is metric-dependent. OCR evaluation was pending at that revision. Current results appear in the executive summary and later sections.


## 10. I6.3a — Final duplicate audit and frozen evaluation samples

**Status:** Completed 25 November 2026, based on Member 1's supplied terminal output. No inference or training performed.

### 10.1 Objective and method

Script: `src/integration/prepare_i6_3_samples.py`. Command: `python src/integration/prepare_i6_3_samples.py` in `.venv-integration` from the repository root.

Read the I6.2 curated and India candidate CSVs; recheck previous DL exclusions. Group rows across both pools by connected components of matching normalized identifiers or matching lowercase/whitespace-normalized ingredient descriptions in any of the three OFF representations (including cross-format matches). Exclude all groups with conflicting recorded NOVA labels. Keep one randomly selected representative from each remaining group, prioritizing India when a group occurs in both pools. With seed 42, draw the natural curated sample first and the balanced sample from the remaining curated representatives. Preserve the India representatives as a separate cohort. This is an exact-key grouping method, not a semantic duplicate detector. A component may link different descriptions transitively.

### 10.2 Grouping and exclusions

| Measurement | Count |
|---|---:|
| Input curated rows | 252,365 |
| Input India rows | 5,782 |
| Total input rows | 258,147 |
| Connected groups | 206,002 |
| Groups with multiple rows | 26,047 |
| Groups shared between cohorts | 65 |
| Conflicting-label groups | 364 |
| Conflicting rows excluded (curated / India) | 1,607 / 41 |
| Nonrepresentative rows excluded (curated / India) | 50,431 / 430 |
| Representative groups retained | 205,638 |
| Cross-sample group overlap | 0 |

The 65 shared groups are not the same measure as the 15 overlapping normalized codes reported in I6.2: the grouping also links exact text across formats. Conflicting labels are excluded, not corrected or adjudicated. Reconciliation: 258,147 − 1,648 − 50,861 = 205,638 = 206,002 − 364.

| NOVA | Curated representative pool | India representative pool | Balanced curated | Natural curated | India evaluation | Unused curated representatives |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 7,335 | 378 | 500 | 364 | 378 | 6,471 |
| 2 | 695 | 53 | 500 | 34 | 53 | 161 |
| 3 | 34,074 | 931 | 500 | 1,688 | 931 | 31,886 |
| 4 | 158,223 | 3,949 | 500 | 7,914 | 3,949 | 149,809 |
| **Total** | **200,327** | **5,311** | **2,000** | **10,000** | **5,311** | **188,327** |

The balanced target was achieved. Natural sample NOVA 4 prevalence is 79.14%, implying a 79.14% accuracy baseline for an always-NOVA-4 classifier; only 34 actual NOVA 2 examples are available. Accuracy must therefore be interpreted with macro F1 and class support. Only 161 unused curated NOVA 2 representatives remain; preserve these groups rather than automatically allocating them to training. These are sampling observations, not model predictions.

### 10.3 Reproducibility and generated outputs

Directory: `data/integration/evaluation_i6_3/`. The sampling report records seed 42, input SHA-256 hashes, sample hashes, all decisions and limitations. Freeze these files; do not regenerate or modify them for subsequent evaluation. The sampling script refuses to overwrite existing outputs.

| File | Observed size | SHA-256 |
|---|---|---|
| balanced_curated.csv | 1.9M | `e1878935db79b0c43050b4ec84fc2663dcda1cd9b880e11a8ac11674fc96a87f` |
| natural_curated.csv | 17M | `92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40` |
| india_evaluation.csv | 3.2M | `c1602b19575634e152fbc2322bc32c29a85c32473ad134fc0e4514cb79bb2c45` |
| i6_3_sampling_report.json | 3.3K | Not supplied |

Hashes are transcribed from the supplied sampling run, not independently calculated on the original data here. The JSON records the upstream file hashes; retain it locally. The three samples have zero reported exact OFF group overlap. Natural sampling is group-weighted within the screened, deduplicated curated pool, not product-weighted retail prevalence. Near duplicates, inaccurate English metadata, and inaccurate OFF labels may remain. USDA descriptions were not part of group formation. The unused groups are not automatically an independent test set if tuning is undertaken.

### 10.4 I6.3b paired experiment design (executed; see Section 11)

Run `HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_3_balanced` with the integration environment active. Verify the frozen balanced sample hash and class counts; load the existing checkpoint once and call the same prediction function on `off_raw`, `off_clean`, and `off_dl_structured` for all 2,000 records (6,000 measured calls, plus warm-up). Record the model file hashes, device and software versions, row-level predictions and confidence scores, class-wise precision/recall/F1/support, accuracy, macro F1, confusion matrices and paired counts (both correct, first only, second only, both wrong). Outputs: `data/integration/evaluation_i6_3b/balanced_predictions.csv` and `i6_3b_balanced_metrics.json`. No new training, tuning or changes to the checkpoint. If the balanced results guide input-format selection, confirm the selected format on a separate frozen cohort. The original structured-text model remains the reference condition. Results are recorded in Section 11.


## 11. I6.3b — Balanced paired model evaluation

**Status:** Completed 25 November 2026, based on Member 1's supplied terminal output. No new training or weight tuning.

### 11.1 Objective and execution

Compare the effect of three ingredient representations using the same frozen 2,000-product cohort (500 products per recorded NOVA class) and the same existing DistilBERT checkpoint. The script verifies sample integrity, loads the model once, warms up the predictor, rotates representation call order across products and makes one prediction per representation per product. Evaluation was run offline with the integration environment active.

```bash
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_3_balanced
```

- Sample: `data/integration/evaluation_i6_3/balanced_curated.csv` (hash in Section 10.3, matched in evaluation report).
- Device: `mps:0`; PyTorch 2.15.0; Transformers 5.24.0.
- Input fields: raw = `off_raw`, clean = `off_clean`, structured = `off_dl_structured`.
- Measured calls: 6,000, excluding warm-up; evaluation elapsed time 78.371 s.

### 11.2 Main performance metrics

| Metric | Original OFF (raw) | BDA comparison-cleaned | DL structured |
|---|---:|---:|---:|
| Accuracy (%) | 88.40 | 60.95 | 89.60 |
| Macro F1 | 0.884713 | 0.556720 | 0.896604 |
| Balanced accuracy (%) | 88.40 | 60.95 | 89.60 |
| Correct / 2,000 | 1,768 | 1,219 | 1,792 |
| Incorrect | 232 | 781 | 208 |
| Mean model confidence (%) | 93.826905 | 84.765525 | 94.174225 |
| Mean predictor call latency (ms) | 13.026749 | 13.031995 | 13.097698 |

Balanced accuracy equals accuracy here because each class has equal support. Confidence is a raw model score, not a calibrated probability of correctness. Timings are local predictor calls, not end-to-end API latency.

### 11.3 Class-wise results

All rows below have support 500; precision, recall and F1 are fractions rounded to four decimals.

| Input | NOVA | Precision | Recall | F1 | Predicted count |
|---|---:|---:|---:|---:|---:|
| Raw | 1 | 0.9254 | 0.9180 | 0.9217 | 496 |
| Raw | 2 | 0.9157 | 0.7820 | 0.8436 | 427 |
| Raw | 3 | 0.7685 | 0.9160 | 0.8358 | 596 |
| Raw | 4 | 0.9563 | 0.9200 | 0.9378 | 481 |
| Clean | 1 | 0.5491 | 0.9060 | 0.6838 | 825 |
| Clean | 2 | 0.5438 | 0.9320 | 0.6868 | 857 |
| Clean | 3 | 0.8267 | 0.1240 | 0.2157 | 75 |
| Clean | 4 | 0.9794 | 0.4760 | 0.6406 | 243 |
| Structured | 1 | 0.9469 | 0.8920 | 0.9186 | 471 |
| Structured | 2 | 0.9004 | 0.8140 | 0.8550 | 452 |
| Structured | 3 | 0.7931 | 0.9200 | 0.8519 | 580 |
| Structured | 4 | 0.9638 | 0.9580 | 0.9609 | 497 |

Confusion matrices: rows = recorded NOVA, columns = predicted NOVA, order 1, 2, 3, 4.

| Input | Actual | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|---|---:|---:|---:|---:|---:|
| Raw | 1 | 459 | 14 | 23 | 4 |
| Raw | 2 | 19 | 391 | 83 | 7 |
| Raw | 3 | 13 | 19 | 458 | 10 |
| Raw | 4 | 5 | 3 | 32 | 460 |
| Clean | 1 | 453 | 45 | 0 | 2 |
| Clean | 2 | 29 | 466 | 4 | 1 |
| Clean | 3 | 192 | 244 | 62 | 2 |
| Clean | 4 | 151 | 102 | 9 | 238 |
| Structured | 1 | 446 | 22 | 29 | 3 |
| Structured | 2 | 12 | 407 | 74 | 7 |
| Structured | 3 | 11 | 21 | 460 | 8 |
| Structured | 4 | 2 | 2 | 17 | 479 |

### 11.4 Paired comparisons

| First → second | Both correct | First only correct | Second only correct | Both wrong | Prediction agreement (%) | Second minus first accuracy (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Raw → clean | 1,120 | 648 | 99 | 133 | 59.65 | -27.45 |
| Raw → structured | 1,731 | 37 | 61 | 171 | 94.90 | +1.20 |
| Clean → structured | 1,126 | 93 | 666 | 115 | 59.70 | +28.65 |

Structured text yields a net 24 additional correct predictions relative to raw. Its advantage is modest; no statistical significance or universal superiority is asserted. Cleaned text yields 549 fewer correct predictions than raw. The clean model inputs show severe NOVA 3 and 4 recall degradation, with 436 actual NOVA 3 and 253 actual NOVA 4 rows predicted as NOVA 1 or 2. High NOVA 2 recall under cleaning reflects many false positives: 466 true positives among 857 predictions. Structured text retains higher NOVA 3 and 4 F1 and higher NOVA 2 F1 than raw, but raw has slightly higher NOVA 1 F1. The original Member 2 balanced result (91.34%, macro F1 0.9132) was on a different 1,501-row sample; the difference cannot be attributed solely to preprocessing.

### 11.5 Interpretation and provisional implementation decision

BDA cleaning serves exact source comparison and quality auditing. It is not the same as DL training preprocessing. Removing punctuation or altering ingredient boundaries may change model tokenization and learned input patterns; this is a hypothesis, not an isolated causal mechanism. No individual cleaning operation was ablated. Keep comparison-cleaned data for BDA matching and retain the original and structured descriptions for inference. Prefer structured OFF text provisionally where available, and raw text for manual entry/OCR where structured data is absent; confirm on a separate frozen cohort. This is a format-selection decision, not a checkpoint change. Do not infer healthfulness from NOVA alone.

### 11.6 Evidence and reproducibility

Output folder `data/integration/evaluation_i6_3b/` contains `balanced_predictions.csv` (302K) and `i6_3b_balanced_metrics.json` (6.2K). SHA-256 of predictions: `66babdee41797d911548772b7caca4b224d207102d7cbf20dbeca6a0aba66343`.

| Model file | SHA-256 |
|---|---|
| config.json | `d3b802f47aed03250232fbfe367bf58e34393bc0831b5853a26a2099268aa829` |
| model.safetensors | `d688e4f77855d2797149caf80006cfd9859b0f92424e7da6d2feb48236cc371b` |
| tokenizer.json | `0d3aef594edd5f9b53e7f814277a9171dc70ff93eb66bda6e01f7aa53997d963` |
| tokenizer_config.json | `e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd` |

Hashes and measurements were transcribed from the supplied run, not independently recalculated on the source files. Labels are OFF references, not independently adjudicated truth. Exact grouping does not rule out near duplicates. Balanced metrics do not estimate natural-distribution precision. All results are conditional on the screened cohort; the model has not been trained on it in this experiment. Preserve the frozen samples and original checkpoint.

## 12. I6.3c — Natural-distribution evaluation (completed)

Evaluate raw and structured OFF text on the already frozen 10,000-row natural curated sample, with the same checkpoint and inference function. Verify the frozen sample hash, class counts and exact group/code separation from balanced and India cohorts; compare checkpoint hashes with I6.3b. Use 20,000 measured predictor calls, with warm-up excluded. The comparison-cleaned format is deliberately omitted from this follow-up based on I6.3b; no claim about its natural-cohort performance is made. This is a post-balanced format selection followed by testing on a separately frozen cohort, not a new model training experiment.

```bash
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_3_natural
```

### 12.1 Execution and results

Run reported on 25 November 2026. All 20,000 measured predictions completed in 270.571867 seconds on `mps:0`, PyTorch 2.15.0, Transformers 5.24.0. The natural sample hash matched Section 10 and all four checkpoint hashes matched Section 11.6. The recorded NOVA supports were 364, 34, 1,688 and 7,914. The always-NOVA-4 baseline scored 79.14% accuracy and 0.220889 macro F1.

| Metric | Original OFF | DL structured |
|---|---:|---:|
| Accuracy (%) | 91.10 | 93.01 |
| Macro F1 | 0.723887 | 0.740930 |
| Balanced accuracy (%) | 86.3620 | 87.5242 |
| Correct / 10,000 | 9,110 | 9,301 |
| Incorrect | 890 | 699 |
| Mean confidence (%) | 95.920210 | 96.778177 |
| Mean call latency (ms) | 13.524761 | 13.517206 |

### 12.2 Class-level metrics

| Input | NOVA | Precision | Recall | F1 | Support | Predicted count |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 1 | 0.7874 | 0.9258 | 0.8510 | 364 | 428 |
| Raw | 2 | 0.1920 | 0.7059 | 0.3019 | 34 | 125 |
| Raw | 3 | 0.6992 | 0.9117 | 0.7915 | 1688 | 2201 |
| Raw | 4 | 0.9950 | 0.9110 | 0.9512 | 7914 | 7246 |
| Structured | 1 | 0.8166 | 0.9176 | 0.8642 | 364 | 409 |
| Structured | 2 | 0.1923 | 0.7353 | 0.3049 | 34 | 130 |
| Structured | 3 | 0.7617 | 0.9129 | 0.8305 | 1688 | 2023 |
| Structured | 4 | 0.9950 | 0.9352 | 0.9642 | 7914 | 7438 |

Confusion matrices (rows recorded OFF NOVA, columns predicted NOVA):

| Input | Actual | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|---|---:|---:|---:|---:|---:|
| Raw | 1 | 337 | 7 | 16 | 4 |
| Raw | 2 | 3 | 24 | 6 | 1 |
| Raw | 3 | 51 | 67 | 1539 | 31 |
| Raw | 4 | 37 | 27 | 640 | 7210 |
| Structured | 1 | 334 | 7 | 20 | 3 |
| Structured | 2 | 3 | 25 | 5 | 1 |
| Structured | 3 | 40 | 74 | 1541 | 33 |
| Structured | 4 | 32 | 24 | 457 | 7401 |

### 12.3 Paired comparison

| Both correct | Raw only correct | Structured only correct | Both wrong | Prediction agreement | Structured minus raw accuracy |
|---:|---:|---:|---:|---:|---:|
| 9036 | 74 | 265 | 625 | 96.45% | +1.91 percentage points |

Structured text gives 191 additional correct predictions, a relative error reduction of 191/890 = 21.46%. NOVA 4 to NOVA 3 errors fall from 640 to 457. The observed advantage of structured over raw is consistent with I6.3b, though raw is necessary when structured input is unavailable. The 93.01% result is not a model improvement over Member 2's 90.11% natural test: the cohorts differ and weights are unchanged.

NOVA 2 remains weak: structured predicts 130 cases, of which 25 are reference-correct, leaving 105 false positives (7 actual NOVA 1, 74 actual NOVA 3, 24 actual NOVA 4); 9 actual NOVA 2 cases are missed. Raw predicts 125, of which 24 are reference-correct. Balanced training versus rare NOVA 2 in the natural cohort is a plausible contributor, not established causation. With only 34 positives, recall is uncertain. Model confidence scores are not calibrated correctness probabilities. Do not tune thresholds or weights using this test cohort. Preserve unused groups and create separate development and untouched evaluation subsets if I7 investigates calibration, prior adjustment or fine-tuning.

### 12.4 Evidence and limitations

- `data/integration/evaluation_i6_3c/natural_predictions.csv` (1.1M), SHA-256 `16393599a553b95c0e602e7e23459cf241021e03263cd28370d656a447eeb2e9`.
- `data/integration/evaluation_i6_3c/i6_3c_natural_metrics.json` (4.8K).
- Natural sample SHA-256 `92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40`.
- Balanced metrics report hash recorded by natural run: `f0d6cde87ea927603603f8b57e4b3511c3ba9dfe4f8be3b02ba161b4a795765c`.
- Source: supplied execution output; underlying files and weights were not independently rerun by report author. OFF reference labels are not independent ground truth; exact screening does not eliminate near duplicates; deduplicated screened sample is not retail market prevalence. Latencies are local predictor calls, not API times. Cleaned input was not repeated. No training or weight tuning occurred.

## 13. I6.4 — OFF–USDA source consistency (completed)

### 13.1 Objective and method

Evaluate whether ingredient text agreement categories created in BDA are associated with agreement in predicted NOVA group. Use raw OFF and raw USDA descriptions for the same matched GTIN, the unchanged DistilBERT predictor and seed 42. Deliberately reuse the I6.3c natural cohort for a descriptive follow-up; this is not independent accuracy validation. Screen previous DL identifiers/text, reserved balanced/India OFF texts, and duplicate selected-pair identifiers/text across both source variants. Select 667 `same_basic`, 667 `same_after_extra_cleanup` and 666 `still_different` pairs, with no cross-stratum backfill. No comparison-cleaned text was fed to the model.

```bash
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_4_sources
```

### 13.2 Selection and execution

| Category | Available before selected-pair deduplication | Target | Selected | Shortfall |
|---|---:|---:|---:|---:|
| same_basic | 4,968 | 667 | 667 | 0 |
| same_after_extra_cleanup | 2,537 | 667 | 667 | 0 |
| still_different | 2,485 | 666 | 666 | 0 |
| Total | 9,990 | 2,000 | 2,000 | 0 |

First-failure screening exclusions: not pair eligible 6, balanced/India exact overlap 2, outside paired categories 2. Four visited rows were rejected for repeated selected-pair text/code during selection; that count is not an exhaustive count of duplicates in the pool. All 4,000 measured calls completed in 54.351092542 seconds on `mps:0`, using PyTorch 2.15.0 and Transformers 5.24.0. Model file hashes match Section 11.6; no training performed.

### 13.3 Agreement outcomes

| Category | Pairs | Agree | Disagree | Agreement | OFF-only accuracy against OFF label | OFF mean confidence | USDA mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| same_basic | 667 | 667 | 0 | 100.00% | 88.91% | 95.07% | 95.16% |
| same_after_extra_cleanup | 667 | 648 | 19 | 97.15% | 90.10% | 96.29% | 96.62% |
| still_different | 666 | 597 | 69 | 89.64% | 91.44% | 96.01% | 96.83% |
| Overall selected pairs | 2,000 | 1,912 | 88 | 95.60% | 90.15% | 95.79% | 96.20% |

All 653 pairs with exact normalized raw-text matches (all in same_basic) agreed. The 69 disagreements in still_different account for 78.41% of the 88 disagreements. Agreement declines across the three categories in this selected sample, supporting the descriptive hypothesis. No causal claim or independent USDA accuracy estimate is made. OFF-only accuracy is a secondary descriptive measure on reused products, not another independent test.

### 13.4 Prediction transition matrices

Rows are OFF predictions; columns are USDA predictions, both ordered NOVA 1–4. Diagonal cells are agreement, not verified correctness.

| Category | OFF predicted | USDA 1 | USDA 2 | USDA 3 | USDA 4 |
|---|---:|---:|---:|---:|---:|
| Overall | 1 | 72 | 3 | 2 | 3 |
| Overall | 2 | 2 | 14 | 4 | 0 |
| Overall | 3 | 1 | 2 | 394 | 48 |
| Overall | 4 | 1 | 0 | 22 | 1432 |
| same_basic | 1 | 37 | 0 | 0 | 0 |
| same_basic | 2 | 0 | 10 | 0 | 0 |
| same_basic | 3 | 0 | 0 | 199 | 0 |
| same_basic | 4 | 0 | 0 | 0 | 421 |
| same_after_extra_cleanup | 1 | 18 | 1 | 0 | 0 |
| same_after_extra_cleanup | 2 | 0 | 3 | 4 | 0 |
| same_after_extra_cleanup | 3 | 0 | 1 | 100 | 10 |
| same_after_extra_cleanup | 4 | 0 | 0 | 3 | 527 |
| still_different | 1 | 17 | 2 | 2 | 3 |
| still_different | 2 | 2 | 1 | 0 | 0 |
| still_different | 3 | 1 | 1 | 95 | 38 |
| still_different | 4 | 1 | 0 | 19 | 484 |

| Category | Recorded OFF NOVA counts (1,2,3,4) | OFF prediction counts | USDA prediction counts |
|---|---|---|---|
| Overall | 64, 4, 337, 1595 | 80, 20, 445, 1455 | 76, 19, 422, 1483 |
| same_basic | 37, 4, 156, 470 | 37, 10, 199, 421 | 37, 10, 199, 421 |
| same_after_extra_cleanup | 12, 0, 79, 576 | 19, 7, 111, 530 | 18, 5, 107, 537 |
| still_different | 15, 0, 102, 549 | 24, 3, 135, 504 | 21, 4, 116, 525 |

Seventy disagreements involve NOVA 3 and 4 (48 OFF 3 → USDA 4; 22 OFF 4 → USDA 3). The sample is dominated by NOVA 4, and has only four recorded NOVA 2 references, all in same_basic. Do not generalize overall agreement to each class.

### 13.5 Implications and limitations

Retain OFF and USDA ingredient descriptions separately with provenance and the existing BDA source-discrepancy flag. Preserve raw text for inference, structured OFF text when available, and comparison-cleaned text for matching. A matching GTIN does not establish identical formulation or time period. A disagreement is not evidence that one particular source is wrong; matching predictions can both be incorrect. USDA lacks independent NOVA reference labels. Text matching categories are not measures of semantic equivalence. English language not independently verified, near duplicates may remain, and confidence is uncalibrated. Categories are deliberately sampled near-equally, so 95.60% is not a population prevalence estimate. I6.4 reuses I6.3c products and should not be counted as independent accuracy evidence.

### 13.6 Evidence and reproducibility

Output folder `data/integration/evaluation_i6_4/`: `source_pairs.csv` (3.5M), `source_predictions.csv` (179K), `i6_4_source_consistency.json` (7.0K). Reported hashes:

| Artifact | SHA-256 |
|---|---|
| Frozen natural sample | `92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40` |
| Natural metrics report | `f592646d63dd4631fbe0ae41c62268443d930e9fc629dd93e41109630152d56c` |
| Selected pairs | `85e8cea1be48c0a782364ca2f8a99e18f49490a4c82d73a5bf3f7f8e22418fca` |
| Predictions | `110f97150f4a50c7999d4d810d9537b5779b8f296e5debf3c0baff76e6f23f5d` |

Results and hashes transcribed from the supplied terminal output. The underlying CSVs and checkpoint have not been independently rerun or hashed by the report author.

## 14. I6.5 — India-tagged evaluation (completed)

### 14.1 Objective and method

Evaluate the entire frozen `india_evaluation.csv` cohort using original OFF ingredient text (`off_raw`) and training-style structured text (`off_dl_structured`) with the unchanged DistilBERT predictor. Raw and structured formats were selected following I6.3b. Compare both formats on identical products, rather than treating the two sets of calls as independent samples. The script reported 5,311 unique codes, 5,311 unique groups, 10,480 unique OFF text keys, zero exact overlap with previous DL samples and zero exact overlap with other evaluation samples. These are script-reported exact-match checks, not proof of absence of near duplicates.

```bash
HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_5_india
```

| Recorded NOVA | Support |
|---|---:|
| 1 | 378 |
| 2 | 53 |
| 3 | 931 |
| 4 | 3,949 |
| Total | 5,311 |

10,622 measured predictions completed in 143.29079441697104 seconds on `mps:0`, PyTorch 2.15.0 and Transformers 5.24.0. The four checkpoint-file hashes match I6.3b (Section 11.6). No training or weight tuning occurred.

### 14.2 Overall metrics

| Metric | Raw OFF | Structured OFF |
|---|---:|---:|
| Accuracy | 79.871964% | 80.060252% |
| Macro F1 | 0.6258021 | 0.6254859 |
| Balanced accuracy | 83.472562% | 82.709901% |
| Correct | 4,242 | 4,252 |
| Incorrect | 1,069 | 1,059 |
| Mean confidence | 91.020921% | 92.627027% |
| Mean call latency | 13.483449 ms | 13.479103 ms |

Always predicting NOVA 4 gives 74.355112% accuracy and 0.21322894 macro F1. Both model formats outperform this baseline. Structured yields ten additional correct classifications (+0.188288 percentage points) but raw yields slightly better macro F1 and balanced accuracy. This does not establish a clear overall format winner for India-tagged products. Structured accuracy is about 12.95 percentage points below the natural curated cohort (93.01%); cohort composition, source quality and selection differ, so geography alone cannot explain the gap.

### 14.3 Class-wise results

Precision, recall and F1 are fractions; support and predicted count are counts.

| Format | NOVA | Precision | Recall | F1 | Support | Predicted count |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 1 | 0.673423 | 0.791005 | 0.727494 | 378 | 444 |
| Raw | 2 | 0.134565 | 0.962264 | 0.236111 | 53 | 379 |
| Raw | 3 | 0.558442 | 0.785177 | 0.652679 | 931 | 1309 |
| Raw | 4 | 0.994338 | 0.800456 | 0.886925 | 3949 | 3179 |
| Structured | 1 | 0.675862 | 0.777778 | 0.723247 | 378 | 435 |
| Structured | 2 | 0.137741 | 0.943396 | 0.240385 | 53 | 363 |
| Structured | 3 | 0.554033 | 0.781955 | 0.648552 | 931 | 1314 |
| Structured | 4 | 0.994061 | 0.805267 | 0.889759 | 3949 | 3199 |

Confusion matrices: rows = recorded OFF NOVA; columns = predicted NOVA.

| Format | Actual | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|---|---:|---:|---:|---:|---:|
| Raw | 1 | 299 | 48 | 27 | 4 |
| Raw | 2 | 1 | 51 | 1 | 0 |
| Raw | 3 | 85 | 101 | 731 | 14 |
| Raw | 4 | 59 | 179 | 550 | 3161 |
| Structured | 1 | 294 | 45 | 34 | 5 |
| Structured | 2 | 0 | 50 | 3 | 0 |
| Structured | 3 | 90 | 99 | 728 | 14 |
| Structured | 4 | 51 | 169 | 549 | 3180 |

Structured NOVA 2: 50 of 363 predicted cases agree with the recorded label (13.77% precision); 313 false positives (45 NOVA 1, 99 NOVA 3, 169 NOVA 4) and three false negatives. Raw: 51 of 379 predicted cases are reference-correct (13.46% precision), with 328 false positives. Only 53 recorded positives are available. The recurring high-recall/low-precision pattern motivates error analysis and prior/calibration experiments on new development data. High NOVA 4 precision does not imply high recall: 769 of 3,949 recorded NOVA 4 products are missed by structured input, including 549 predicted NOVA 3 and 169 predicted NOVA 2.

### 14.4 Paired comparisons

| Both correct | Raw only correct | Structured only correct | Both wrong | Prediction agreement | Structured minus raw accuracy |
|---:|---:|---:|---:|---:|---:|
| 3975 | 267 | 277 | 792 | 88.853323% | +0.188288 pp |

The small net difference hides substantial per-product changes. Structured remains the provisional OFF default because of the balanced and natural curated results, with raw as fallback; the India result cautions against claiming universal superiority. Do not choose a new India-specific rule using this already-inspected cohort and then claim unbiased validation on the same cohort.

### 14.5 Evidence and limitations

- Output directory: `data/integration/evaluation_i6_5/`.
- `india_predictions.csv` (602K as displayed by macOS); `i6_5_india_metrics.json` (5.4K).
- Sample SHA-256: `c1602b19575634e152fbc2322bc32c29a85c32473ad134fc0e4514cb79bb2c45`.
- Predictions SHA-256: `90888090edc0999821859f894ea993aae46bcdab427eb9c0d9441e2e06c3d07e`.
- Natural report SHA-256: `f592646d63dd4631fbe0ae41c62268443d930e9fc629dd93e41109630152d56c`.
- Source report SHA-256: `41daebe178fc0b67e96511674f181c096bedb40408df34c717218d88be1c66ec`.

All numerical findings and hashes are from the supplied terminal output; the underlying CSVs and checkpoint were not independently rerun by the report author. India tagging does not establish manufacture in India or representativeness of Indian retail. English metadata does not establish English text, and OFF labels are not independently verified. Exact screening does not rule out near-duplicate formulations. Confidence is uncalibrated; latency is not end-to-end API latency. Cohorts have different selection mechanisms and must not be averaged into a single project accuracy. No threshold or weight tuning was performed.

## 15. I6.6 — OCR-to-DL pilot (protocol and results)

Inventory the existing OCR outputs, matching image references and human transcriptions before defining the paired evaluation. The BDA report documents two manually transcribed whole-image references (not ingredient-only gold standard) and a separate eight-image balanced OCR pilot with two segmentation modes, but the local raw evidence has not yet been inspected for I6.6. Do not treat whole-package text or OFF database descriptions as verified ingredient-only image transcriptions. Where feasible, prepare ingredient-only references from the same photographed label and compare the unchanged model's raw-input predictions with OCR text from the corresponding image. Report per-image word error rate and prediction changes, with the reference-text prediction treated as a comparison, not independent NOVA truth. Document prior-experiment overlaps and any image/label ambiguities. A tiny pilot cannot establish OCR or NOVA generalization accuracy; Tesseract findings do not directly validate Android ML Kit. No new OCR evaluation results yet.

Revision 0.6: 25 November 2026 — completed I6.5, preserved earlier findings and documented I6.6 preparation.

### 15.1 Inventory review (completed; inference pending)

The supplied local terminal listing confirms six original ingredient-image files, two existing manual transcription files (`8902979052405.txt`, `8903553002809.txt`), 12 enhanced OCR records, and 16 balanced OCR records from eight images. The balanced OCR JSONL includes barcode, recorded NOVA, PSM, manual ingredient/readability flags, original and prepared image paths, OCR text, status and processing time. The earlier `run_balanced_ocr.py` resizes images to at most 3,000 pixels on the longest side and runs English Tesseract PSM 6 and 11 on the same prepared full image. The first supplied OCR record includes nutrition information (protein, sugars, cholesterol, sodium), demonstrating that these outputs are not guaranteed to be ingredient-only. The image pixels and contents of the two existing reference files have not yet been reviewed for I6.6.

| Recorded NOVA | Balanced image barcodes |
|---|---|
| 1 | 8904083301394, 89080955 |
| 2 | 8901088000468, 9556023331597 |
| 3 | 8906112665354, 0760851638226 |
| 4 | 8901063367258, 8901414006850 |

### 15.2 Planned controlled ingredient-region pilot

Inspect all eight original photos before inference. Include only images with an identifiable, sufficiently readable English ingredient section. Record exclusions and reasons; do not fill missing words using the OFF database or guess illegible text. Preserve original images, then crop the ingredient section once per eligible photo and create a human transcription of that same region, preserving ingredient order and separators as visible. Freeze references before reviewing OCR results or model predictions. Run Tesseract PSM 6 and 11 on the identical crop, using English recognition and recording version/settings and hashes. Retain empty/error OCR attempts as failures and report the denominator; do not feed empty strings into the predictor. Use the unchanged checkpoint verified in I6.3b. Calculate per-image and pooled word error rate using the earlier BDA word normalization, plus prediction agreement and confidence changes against the human-transcription prediction. Report recorded OFF label agreement only as descriptive context, not independent truth. Check barcode/text overlap with previous DL/evaluation samples and disclose overlaps; this is a small robustness pilot rather than independent generalization testing. Target at most eight images and 24 model calls (one human text and two OCR outputs per image). Full-package OCR may be evaluated separately as a workflow stress test; do not conflate extra packaging text with recognition errors. Tesseract results cannot validate Android ML Kit directly.

Revision 0.6.1: inventory and protocol recorded; no new OCR predictions, WER measurements, or model training performed.


### 15.3 Image inspection and reference preparation (I6.6.4)

The supplied eight original photographs in `I6_6_Ingredient_Images.zip` were visually inspected, including magnified views. Six were retained for the pilot. Two were excluded before any new OCR or model predictions: `8901414006850` (ingredient text clipped at left edge) and `9556023331597` (fold obscures an additive code; complete transcription uncertain). Retention is 6/8; recorded NOVA supports are 2, 1, 2, 1. This is no longer balanced, and these recorded classes are not independently validated. The selected readable images cannot represent all label photographs.

| Barcode | Recorded NOVA | Preparation |
|---|---:|---|
| 8904083301394 | 1 | Milk and milk solids, isolated ingredient line |
| 89080955 | 1 | Coffee beans and chicory, polygon crop |
| 8901088000468 | 2 | Coconut oil, isolated ingredient line |
| 0760851638226 | 3 | Lobia/millet pancake ingredient list, heading excluded because it contains product description |
| 8906112665354 | 3 | Seeds, jaggery, nuts, oat flour and honey, ingredient-only image |
| 8901063367258 | 4 | Biscuit ingredient paragraph, excluding explanatory and allergen statements |

AI-assisted visual transcriptions were drafted from original image pixels, not copied from OFF ingredient descriptions or derived from OCR outputs. User visual verification is required before running the experiment. These are provisional references, not independently human-verified yet. Images were EXIF-oriented, polygon-masked to white outside the selected region, cropped to bounding box and padded by 20 pixels; no enhancement or resizing of retained pixels. Crop and reference SHA-256 hashes, original image hashes, polygon coordinates and exclusions are recorded in `data/integration/evaluation_i6_6_inputs/manifest.json`. Ingredient headings are retained when visible within the chosen region and are present in the corresponding reference; this is a controlled raw-input robustness test, not a standardized ingredient parser.

The prepared `I6_6_OCR_Evaluation_Pack.zip` contains six PNG crops, six provisional reference TXT files, the manifest and `src/integration/evaluate_i6_6_ocr.py`. The script checks original image and prepared input hashes, earlier sample hashes, checkpoint equality with I6.5, and reports exact identifier/text matches against previous datasets and evaluation cohorts. Overlaps are disclosed rather than excluded because this is a paired robustness pilot. After visual reference review, run `HF_HUB_OFFLINE=1 python -m src.integration.evaluate_i6_6_ocr --references-reviewed`. Tesseract PSM 6 and 11 use the same crop with English recognition; up to 18 inference calls (six reference plus twelve OCR predictions). Word error rate ignores case/punctuation using the earlier tokenization; empty successful OCR outputs count as 100% WER, engine failures are reported separately and excluded from pooled WER. Prediction agreement is reported both among nonempty predictions and across all attempts, using the reference-text prediction as baseline, not independent NOVA truth. The output directory is `data/integration/evaluation_i6_6/`.

Script syntax and synthetic edit-distance, overlap and summary tests passed. Actual OCR and model inference have not yet been executed. No new WER or classification findings are available. The script requires the existing Tesseract executable and English language data; no new Python packages are intended. The pilot does not test automatic region selection or Android ML Kit.

Revision 0.6.2: image inspection, exclusions, provisional references and controlled experiment package prepared; user review and inference pending.


### 15.4 Completed OCR robustness experiment (I6.6.5)

The supplied terminal output confirms the `--references-reviewed` run completed on MPS using Tesseract 5.5.4 (`eng`), torch 2.15.0 and transformers 5.24.0. The review flag records the user's confirmation, not a separately observed independent transcription review. The same manually isolated, unenhanced ingredient crops were used for both PSM modes. Six reference predictions and eight nonempty OCR predictions were made (14 calls); four empty outputs were not passed to the model. Reported evaluation wall time was 1.568439 seconds, not an API or mobile latency benchmark. Both modes had zero engine errors.

| Metric | PSM 6 | PSM 11 |
|---|---:|---:|
| Attempts | 6 | 6 |
| Nonempty outputs / predictions | 6 | 2 |
| Empty outputs | 0 | 4 |
| Reference words in successful engine runs | 95 | 95 |
| Word edit errors | 61 | 68 |
| Pooled WER | 64.210526% | 71.578947% |
| Reference-prediction agreements | 3 | 2 |
| Agreement among nonempty predictions | 50.00% | 100.00% |
| Agreement across all attempts | 50.00% | 33.333333% |
| Mean change in top-score confidence among predictions | -12.521667 pp | -10.79 pp |

The 100% conditional PSM 11 agreement is based on only two outputs and does not include its four empty results. Word error rates include successful empty outputs as 100% WER and ignore case and punctuation. The pooled denominator weights longer texts more heavily; punctuation-only output can be nonempty while having zero word tokens.

| Barcode | Recorded OFF | Reference prediction | PSM | OCR words | WER (%) | OCR prediction | Reference score (%) | OCR score (%) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0760851638226 | 3 | 3 | 6 | 1 | 100.00 | 1 | 93.13 | 90.8 |
| 0760851638226 | 3 | 3 | 11 | 0 | 100.00 | None | 93.13 | — |
| 8901063367258 | 4 | 3 | 6 | 0 | 100.00 | 1 | 89.36 | 80.0 |
| 8901063367258 | 4 | 3 | 11 | 0 | 100.00 | None | 89.36 | — |
| 8901088000468 | 2 | 2 | 6 | 3 | 0.00 | 2 | 89.87 | 84.63 |
| 8901088000468 | 2 | 2 | 11 | 0 | 100.00 | None | 89.87 | — |
| 8904083301394 | 1 | 1 | 6 | 4 | 0.00 | 1 | 81.67 | 59.91 |
| 8904083301394 | 1 | 1 | 11 | 0 | 100.00 | None | 81.67 | — |
| 8906112665354 | 3 | 3 | 6 | 26 | 0.00 | 3 | 61.32 | 61.32 |
| 8906112665354 | 3 | 3 | 11 | 26 | 0.00 | 3 | 61.32 | 61.32 |
| 89080955 | 1 | 1 | 6 | 1 | 80.00 | 2 | 99.12 | 62.68 |
| 89080955 | 1 | 1 | 11 | 2 | 80.00 | 1 | 99.12 | 77.54 |


**Failure examples.** `0760851638226`: PSM 6 produced `aoa` (100% WER), changing NOVA 3 (93.13% score) to NOVA 1 (90.80%). `8901063367258`: PSM 6 produced `=` (100% WER, zero word tokens), changing NOVA 3 (89.36%) to NOVA 1 (80.00%). For this second image the recorded OFF label was NOVA 4; reference-text predictions are not gold labels. `89080955`: PSM 6 returned just `Ingredients`, while PSM 11 returned `ingredients\n\nAM`; both had 80% WER, but the latter happened to agree with the reference prediction. Thus agreement does not establish usable OCR or correct classification. Milk text had zero word errors in PSM 6 but a 21.76-point drop in confidence, plausibly related to punctuation differences; WER ignores punctuation. The exact cause was not isolated. The seeds/jaggery product had 0% WER and identical model scores for both modes.

**Overlap:** `0760851638226`, `8901063367258`, and `8906112665354` matched the reserved India evaluation by barcode. No exact reference/OCR text match was reported. These overlaps were disclosed rather than excluded as specified in the robustness protocol.

**Implications:** require review and editing of OCR text, reject empty/punctuation-only/heading-only text before inference, prompt for retake or manual entry when extraction is suspicious. Do not impose a blanket minimum number of words because real products can have a single ingredient. Meaningless alphabetic fragments require further checks or human review; confidence cannot be the sole quality gate. The proposed guards are not yet implemented or validated. Keep raw ingredient text rather than comparison-cleaned text.

**Limitations:** six selected readable images (recorded NOVA supports 2/1/2/1), two exclusions before inference; manual cropping does not test automatic ingredient-region detection; AI-assisted references reviewed according to run flag; near duplicates not screened; uncalibrated confidence; no independent NOVA truth; Tesseract results are not Android ML Kit validation. This is not a representative OCR benchmark.

**Evidence and reproducibility:** `data/integration/evaluation_i6_6/i6_6_ocr_metrics.json` and `ocr_predictions.jsonl` (17K and 11K in supplied listing). All reported results from local terminal output; full datasets not rerun by report author.

- `manifest_sha256`: `e2cdbba385c1a00309992d95a79dcb68379ce9af26b903844cdda6e060ff7c98`.
- `india_report_sha256`: `b6b85fccdaa48b35850f04bbd1556d664e88276291d19fd310248860deda6235`.
- `predictions_sha256`: `3667855e5a61202631acf028518e8f0a310fc818a879c7a56afe6d26e042130f`.
- Checkpoint and tokenizer hashes match the previous I6.3b–I6.5 reports.


## 16. I6 conclusion and I7 plan

I6 is complete. The selected unchanged model is a workable baseline, not a fully validated production classifier. Structured OFF inputs performed best in both curated experiments; original text is the practical fallback; India results show negligible accuracy gain and slightly worse macro F1 for structured text. Comparison-cleaned text remains a source-matching representation, not an inference input. OFF–USDA prediction agreement is not USDA accuracy. NOVA 2 precision remains low on naturally imbalanced data, and OCR can generate confident predictions from unusable input. These findings motivate targeted improvement rather than automatic full retraining.

| Cohort | Structured accuracy | Structured macro F1 | NOVA 2 precision |
|---|---:|---:|---:|
| Balanced curated | 89.60% | 0.8966 | 90.04% |
| Natural curated | 93.01% | 0.7409 | 19.23% |
| India-tagged | 80.06% | 0.6255 | 13.77% |

### I7.1 — Original diagnostic error analysis protocol (completed; results in Section 17)

Read existing frozen natural and India samples and prediction CSVs, verify hashes and join identities, and recompute confusion matrices. Describe score bins and high-confidence errors for raw and structured formats, NOVA 2 false-positive origins, and NOVA 4 to 3/2 errors. Export a fixed-seed (42), error-enriched structured-input review sample, up to ten rows per diagnostic category and cohort, with fields for manual observations and possible label ambiguity. This is exploratory analysis of already-seen evaluation data, not new inference, threshold fitting, or independent validation. Do not claim observed labels are erroneous without verification.

### I7.2 onward — Planned, not executed

Reconstruct complete connected exact-text/code groups from the audited pools and preserve separation from all used groups, including OCR and source-pair texts/codes. Recheck the earlier balanced `training_sample.csv` against its three partitions if not already verified. Create new development and untouched final test sets from eligible unused groups before tuning; existing 188,327 unused curated representatives are only a starting count. If new India-tagged evaluation data are unavailable, state that limitation rather than tune and retest on I6.5. Compare unchanged baseline with simple prior adjustment and/or calibration on development data, with class-wise precision/recall, macro F1, and coverage if abstention is considered. Freeze the decision before testing. Prior adjustment assumes a form of label shift and must be empirically tested, not inferred solely from balanced training. Use fine-tuning only if justified and keep the original checkpoint. No model retraining or decision threshold has been applied yet.

Revision 0.7: 25 November 2026 — I6.6 results and I6 conclusion recorded; I7.1 diagnostic protocol added. Previous revisions' pending statements are historical.


## 17. I7.1 — Completed descriptive error audit

Executed `python -m src.integration.analyze_i7_errors` on 25 November 2026. The supplied terminal report shows successful hash-checked joins and recomputed confusion matrices for both formats on 10,000 natural curated and 5,311 India-tagged products. Total saved predictions analyzed: 30,622; no new model calls, model fitting or threshold selection. The 129-row review sample is intentionally error-enriched and awaits qualitative inspection; no reference labels have been independently corrected.

### 17.1 Overall diagnostics

| Cohort / format | Correct / total | Accuracy | Mean score | Errors ≥90% score | NOVA 2 TP / FP / FN | NOVA 2 FP ≥90% |
|---|---:|---:|---:|---:|---:|---:|
| natural / raw | 9,110/10,000 | 91.10% | 95.92% | 519 | 24 / 101 / 10 | 30 |
| natural / structured | 9,301/10,000 | 93.01% | 96.78% | 421 | 25 / 105 / 9 | 32 |
| india / raw | 4,242/5,311 | 79.87% | 91.02% | 455 | 51 / 328 / 2 | 124 |
| india / structured | 4,252/5,311 | 80.06% | 92.63% | 485 | 50 / 313 / 3 | 127 |

### 17.2 Confidence-bin findings

Bins are lower-inclusive and upper-exclusive, except 100% is included in the final bin. Accuracy is agreement with recorded OFF labels, not an independent truth. Mean confidence is a raw model score, not calibrated probability.

**natural / raw**

| Score bin | N | Correct | Accuracy (%) | Mean score (%) | NOVA 2 predicted N | NOVA 2 correct |
|---|---:|---:|---:|---:|---:|---:|
| 0-50 | 46 | 18 | 39.13 | 45.70 | 8 | 0 |
| 50-70 | 296 | 165 | 55.74 | 60.51 | 26 | 1 |
| 70-80 | 268 | 177 | 66.04 | 75.38 | 22 | 1 |
| 80-90 | 427 | 306 | 71.66 | 85.39 | 20 | 3 |
| 90-95 | 559 | 438 | 78.35 | 92.86 | 23 | 5 |
| 95-100 | 8404 | 8006 | 95.26 | 98.84 | 26 | 14 |

**natural / structured**

| Score bin | N | Correct | Accuracy (%) | Mean score (%) | NOVA 2 predicted N | NOVA 2 correct |
|---|---:|---:|---:|---:|---:|---:|
| 0-50 | 29 | 18 | 62.07 | 45.83 | 3 | 0 |
| 50-70 | 233 | 129 | 55.36 | 60.91 | 28 | 0 |
| 70-80 | 188 | 110 | 58.51 | 75.63 | 19 | 2 |
| 80-90 | 338 | 253 | 74.85 | 85.57 | 27 | 2 |
| 90-95 | 438 | 333 | 76.03 | 92.88 | 18 | 2 |
| 95-100 | 8774 | 8458 | 96.40 | 98.98 | 35 | 19 |

**india / raw**

| Score bin | N | Correct | Accuracy (%) | Mean score (%) | NOVA 2 predicted N | NOVA 2 correct |
|---|---:|---:|---:|---:|---:|---:|
| 0-50 | 119 | 45 | 37.82 | 44.23 | 32 | 0 |
| 50-70 | 408 | 186 | 45.59 | 60.50 | 82 | 1 |
| 70-80 | 299 | 152 | 50.84 | 75.27 | 59 | 2 |
| 80-90 | 493 | 322 | 65.31 | 85.74 | 38 | 4 |
| 90-95 | 648 | 455 | 70.22 | 92.86 | 47 | 3 |
| 95-100 | 3344 | 3082 | 92.17 | 98.24 | 121 | 41 |

**india / structured**

| Score bin | N | Correct | Accuracy (%) | Mean score (%) | NOVA 2 predicted N | NOVA 2 correct |
|---|---:|---:|---:|---:|---:|---:|
| 0-50 | 77 | 22 | 28.57 | 44.78 | 19 | 0 |
| 50-70 | 359 | 148 | 41.23 | 60.40 | 75 | 2 |
| 70-80 | 241 | 116 | 48.13 | 75.05 | 44 | 1 |
| 80-90 | 396 | 213 | 53.79 | 85.80 | 54 | 3 |
| 90-95 | 517 | 336 | 64.99 | 93.00 | 38 | 1 |
| 95-100 | 3721 | 3417 | 91.83 | 98.54 | 133 | 43 |

### 17.3 NOVA 2 false positives and major transitions

| Cohort / format | Actual 1 → 2 | Actual 3 → 2 | Actual 4 → 2 | Actual 4 → 3 | Mean score, NOVA 2 FP | Mean score, NOVA 2 TP |
|---|---:|---:|---:|---:|---:|---:|
| natural / raw | 7 | 67 | 27 | 640 | 76.43% | 93.30% |
| natural / structured | 7 | 74 | 24 | 457 | 78.59% | 94.33% |
| india / raw | 48 | 101 | 179 | 550 | 77.13% | 95.31% |
| india / structured | 45 | 99 | 169 | 549 | 79.95% | 95.33% |

For structured text, 421/699 natural errors (60.23%) and 485/1,059 India errors (45.80%) had score ≥90%. The overall score minus accuracy gap is 3.77 pp for natural and 12.57 pp for India; this is a descriptive gap, not a full calibration metric. NOVA 2 predictions with score ≥95% had 19/35 (54.29%) precision and 19/34 (55.88%) recall on natural, and 43/133 (32.33%) precision and 43/53 (81.13%) recall on India. These are descriptive calculations, not approved thresholds. High-confidence errors remain.

### 17.4 Evidence and limitations

Files: `data/integration/evaluation_i7_1/i7_1_error_audit.json` (16K) and `error_review_sample.csv` (31K). Review seed 42; 129 rows. Review SHA-256: `ab96b01046c561d40f2faa2160452bc12e3b2050f853474b3525ffc269077939`.

- natural: sample_sha256 `92936831b02322f4a91d216108eb632be36c507585b52e9d241bb639b30b4f40`; predictions_sha256 `16393599a553b95c0e602e7e23459cf241021e03263cd28370d656a447eeb2e9`; report_sha256 `f592646d63dd4631fbe0ae41c62268443d930e9fc629dd93e41109630152d56c`.
- india: sample_sha256 `c1602b19575634e152fbc2322bc32c29a85c32473ad134fc0e4514cb79bb2c45`; predictions_sha256 `90888090edc0999821859f894ea993aae46bcdab427eb9c0d9441e2e06c3d07e`; report_sha256 `b6b85fccdaa48b35850f04bbd1556d664e88276291d19fd310248860deda6235`.

Limitations: already-inspected samples, unverified OFF reference labels, class imbalance and cohort differences, no new independent inference; confidence bins are descriptive and some have small support. The error-enriched review sample cannot estimate population prevalence. The actual ingredient examples have not yet been reviewed by the report author.

## 18. I7.2 — Fresh development and holdout preparation (planned)

Reconstruct connected exact normalized OFF code/text groups from the original I6.2 pools and validate I6.3a source hashes, group counts and prior sample identifiers. Include `training_sample.csv` in the exclusion union and compare its full row multiset with the union of train/validation/test. Reserve earlier samples, every India-containing group, selected OFF–USDA pair codes and texts, and all OCR image codes plus reference/recognized text. Drop conflicting-label groups. Choose one random representative per remaining group, and draw two disjoint stratified samples with seed 20261125.

| NOVA class | Target development | Target holdout |
|---|---:|---:|
| 1 | 500 | 500 |
| 2 | Up to 80 | Up to 80 |
| 3 | 1,000 | 1,000 |
| 4 | 3,500 | 3,500 |
| Total | Up to 5,080 | Up to 5,080 |

Actual quotas are capped at half the available groups per class. The previous unused curated pool had 161 NOVA 2 representatives before these additional exclusions, so the quota is not guaranteed. The strata deliberately enrich rare classes; report both unweighted metrics and separately standardized estimates using the original screened curated representative distribution (not market prevalence). Record the remaining eligible group counts and class weights. Never inspect holdout predictions or select a method based on them before freezing development decisions. No fresh India test set is currently available; no claim of India improvement can be established from I6.5 reuse. Near duplicates and label errors can remain.

Revision 0.8 — I7.1 completed and I7.2 design recorded. I7.2 has not been executed. Previous revision statements are historical.


## 19. I7.2 — Development and holdout sampling results (executed)

On 25 November 2026, the supplied terminal output confirmed that the I7.2 sampler completed without inference or tuning. Seed: 20261125. It reconstructed 206,002 connected exact OFF groups, screened both prior candidate pools (252,365 curated and 5,782 India), and reserved earlier evaluation groups, all India groups, original DL files including the full training sample, source-pair text/code, and OCR references/output and image codes. Counts for reservation reasons may overlap. First-failure exclusions: 12,078 previously used/linked curated groups, 5,311 groups containing India, 364 conflicting-label groups.

| NOVA | Eligible unused groups | Development | Holdout | Remaining unused |
|---|---:|---:|---:|---:|
| 1 | 6,470 | 500 | 500 | 5,470 |
| 2 | 161 | 80 | 80 | 1 |
| 3 | 31,878 | 1,000 | 1,000 | 29,878 |
| 4 | 149,740 | 3,500 | 3,500 | 142,740 |
| Total | 188,249 | 5,080 | 5,080 | 178,089 |

All targets reached; zero development/holdout group overlap. Both samples deliberately enrich NOVA 2. Reference class priors (original screened curated representative pool): 0.0366151343, 0.0034693276, 0.1700918997, 0.7898236384. Standardization weights for classes 1–4 in both splits: 0.3720097640, 0.2203023057, 0.8640668507, 1.1463725951. These are distribution-standardized estimates, not retail prevalence; depletion and screening can change within-class representativeness. Report both unweighted and standardized metrics.

Evidence: `data/integration/evaluation_i7_2/i7_2_holdout_report.json`; `development.csv` SHA-256 `0fc4c45c2c4bba6c3910f83da2e30ea304deaa6552c382bb06a3aaf38b626c6a`; `holdout.csv` SHA-256 `6523cfa3fb11bc91df1d8243d4fe11e51e0f92a0cff29a0e0398f3766a6b24ea`. Keep both files unchanged; do not infer on holdout before freezing adjustments.

### 19.1 Unexpected DL artifact provenance discrepancy

`training_sample.csv` contains 10,000 rows; train/validation/test union also contains 10,000. Their exact (code, ingredient_text, nova_group) row multisets differ: 7,501 rows exclusive to each side, implying 2,499 exact shared rows counted with multiplicity. This is not evidence by itself of leakage, a different checkpoint, or 7,501 different products: text changes can cause row differences. Both artifact sets were included in I7.2 exclusion screening; no old-DL-linked groups were reported in the reserved groups dictionary. This is reassuring for exact screening against the supplied artifacts, but it does not establish the checkpoint's training provenance. Earlier I6 audits did not explicitly include training_sample.csv. A read-only follow-up will compare normalized codes/text/rows, check earlier samples against the full training sample, inspect the splitting source, and seek confirmation of the actual checkpoint training inputs from Member 2. No files should be regenerated and no existing results should be silently discarded or upgraded to independent validation.

## 20. I7.1 qualitative review — selected observations (executed)

The uploaded error-review CSV has 129 rows (66 natural, 63 India), SHA-256 `ab96b01046c561d40f2faa2160452bc12e3b2050f853474b3525ffc269077939`, matching the I7.1 output. It is an error-enriched selection with correctly classified controls, not a random prevalence study. Examples below are descriptive and not independently adjudicated NOVA labels.

| Cohort | Recorded → predicted | Score | Observed ingredient text / issue |
|---|---|---:|---|
| India | 1 → 2 | 81.48% | Besan; short regional term, context missing |
| India | 1 → 2 | 94.07% | Mustard, Rai Big; regional naming |
| India | 1 → 2 | 97.78% | Cold Pressed Groundnut Oil, vitamins, vitamin A, vitamin D; recorded label merits review |
| Natural | 2 → 1 | 98.60% | pecans, cashews; recorded label merits review |
| India | 4 → 2 | 78.27% | Sugar, Cocoa Butter, Milk, e322, e476, nature identical flavouring, vanilla; possible missed formulation cues |
| India | 4 → 3 | 95.99% | Chicken/vegetables, TVP, isolate soya protein, e451; possible missed industrial ingredient cues |
| India | 4 → 2 | 72.43% | Nuts, salt, gum arabic, cane sugar plus best-before/storage text; contaminated ingredient field |
| India | 4 → 2 | 97.89% | Edible Common Salt, Potassium iodate, e551, e536; reference classification warrants product-context review |

Correct India NOVA 4 controls include explicit flavourings and industrial additives, while correct NOVA 2 controls include oils and salted butter. Thus additive codes or regional words alone do not explain errors. Observations suggest several nonexclusive hypotheses: regional vocabulary, incomplete/noisy text, ambiguous or questionable reference labels, and missed processing-related cues. Product context and independent expert review are needed before relabeling. No prevalence estimates or causal conclusions are supported by the selected examples. No labels or model weights changed.

Revision 0.9 — I7.2 sampling recorded, qualitative review examples added, provenance discrepancy flagged. I7.3 fitting deferred pending read-only provenance check. Earlier revision statements are historical.


## 21. I7.2b — Read-only DL dataset provenance audit (executed)

The supplied terminal audit compared the complete original `training_sample.csv` (10,000 rows) with the union of train, validation and test (10,000 rows). Results:

| Comparison key | Unique keys per side | Shared unique keys | Exclusive rows per side | Equal multisets |
|---|---:|---:|---:|---|
| Exact code, text, label row | 10,000 | 2,499 | 7,501 | No |
| Normalized code, text, integer label row | 10,000 | 10,000 | 0 | Yes |
| Normalized code | 10,000 | 10,000 | 0 | Yes |
| Normalized ingredient text | 7,169 | 7,169 | 0 | Yes |

The audit uses conservative numeric left-zero-padding to 14 digits, trimming, lowercase and whitespace collapse, and integer NOVA labels. The difference is therefore in representations under these rules, not a missing set of normalized products or labels. The exact formatting transformation responsible is not isolated. The supplied `split_dataset.py` uses `GroupShuffleSplit` over normalized ingredient descriptions (500 candidates, seeds 42 and 43, target 15% test and 15% validation overall), choosing splits based on class proportions and size. The script writes copies of original dataframe rows; its grouping normalization does not explicitly rewrite the ingredient column. Pandas parsing or serialization may contribute to the discrepancy, but this is not established.

| Prior evaluation cohort | Rows | Normalized code overlaps with full sample | Any checked OFF text overlap |
|---|---:|---:|---:|
| Balanced curated | 2,000 | 0 | 0 |
| Natural curated | 10,000 | 0 | 0 |
| India evaluation | 5,311 | 0 | 0 |
| Total | 17,311 | 0 | 0 |

No rerun is indicated by this audit. It confirms consistency of the supplied artifacts and exact screening against the full sample, not independent proof of checkpoint training lineage or absence of near duplicates. The actual training partition counts are 1,742 / 1,758 / 1,767 / 1,751 (NOVA 1–4; total 7,018), and will be used as source priors for I7.3. No files or model weights changed. Source: user-supplied terminal output and splitting code, 25 November 2026.

## 22. I7.3 — Development-only output adjustment (planned, not executed)

Use `development.csv` (5,080 rows) and training-style `off_dl_structured` only. Check the known development, training and original checkpoint hashes. Read the holdout hash from the I7.2 manifest only; do not open or infer on the holdout file. Extract full-precision four-class logits using the existing loaded checkpoint and tokenizer (maximum 256 tokens, truncation, maximum-length padding); compare predictions and probabilities against the existing predictor on 20 development records before proceeding. Cache logits to avoid repeatedly invoking the model. No optimizer updates or weight training.

For logits z, training class proportions q, target screened curated group proportions p, prior strength alpha and positive temperature T, compute `softmax((z + alpha * log(p/q)) / T)`. Apply class bias before temperature; for fixed alpha the positive temperature preserves the argmax. Alpha=0, T=1 is the unchanged baseline; alpha=0 with fitted T is temperature-only. Target priors are not retail or India priors. Prior adjustment is an empirical experiment motivated by label shift, not guaranteed to correct domain shift.

Predeclared design decisions (assistant proposed): evaluate alpha 0.0 to 1.0 in increments of 0.1; for each fit temperature on development by minimizing prior-standardized negative log-likelihood, bounded to [0.25,5], retaining T=1 if improvement is <=0.0001. Select the candidate maximizing prior-standardized macro F1, with ties preferring smaller alpha. Require standardized accuracy within 0.5 percentage points of baseline or better and NOVA 2 recall within 10 percentage points of baseline or better. Require at least 0.005 standardized macro-F1 gain to select nonzero alpha; otherwise retain temperature-only. These are engineering selection constraints, not validated product safety thresholds. Report unweighted and prior-standardized accuracy, macro F1, balanced accuracy, per-class metrics, confusion matrices, NLL, multiclass Brier and 10-bin ECE. Development metrics are fitted/selected and cannot establish generalization.

Save `development_logits.csv`, `i7_3_development_metrics.json` and `frozen_adjustment.json` under `data/integration/evaluation_i7_3/`, including hashes, selection rule, parameters and model identity. The selected configuration remains provisional until I7.4 evaluates it once on the untouched curated holdout. No claim of India-specific improvement or raw/OCR-input calibration can be made. Do not tune on holdout results; if the adjustment fails, retain the baseline and obtain new evaluation data before further selection. Near duplicates, label quality and class-dependent depletion remain limitations.

Revision 0.10 — I7.2b findings recorded; I7.3 design documented, execution pending. Previous revision statements are historical.


## 23. I7.3 — Development adjustment results (completed)

**Source:** Member 1 terminal output, 26 November 2026 local time. All 5,080 structured-text development rows processed on MPS:0; elapsed 65.2538 s for logit collection, with 20 original-predictor parity checks. Torch 2.15.0, Transformers 5.24.0. The script verified the checkpoint and input hashes before inference. No model weights changed; holdout not opened or predicted.

### 23.1 Results

**Unweighted metrics**

| Metric | Original | Temperature only | Combined |
|---|---:|---:|---:|
| accuracy_pct | 92.716535 | 92.716535 | 93.011811 |
| macro_f1 | 0.808678 | 0.808678 | 0.809454 |
| balanced_accuracy_pct | 86.425000 | 86.425000 | 84.416071 |
| nll | 0.264263 | 0.246183 | 0.213386 |
| multiclass_brier | 0.122597 | 0.118453 | 0.108146 |
| ece_10_equal_width | 0.035906 | 0.010843 | 0.012874 |
| mean_confidence_pct | 96.281272 | 93.475499 | 94.299236 |
| NOVA 2 precision | 0.402778 | 0.402778 | 0.435897 |
| NOVA 2 recall | 0.725000 | 0.725000 | 0.637500 |
| NOVA 2 f1 | 0.517857 | 0.517857 | 0.517766 |

**Prior standardized metrics**

| Metric | Original | Temperature only | Combined |
|---|---:|---:|---:|
| accuracy_pct | 93.486884 | 93.486884 | 93.950132 |
| macro_f1 | 0.728895 | 0.728895 | 0.736114 |
| balanced_accuracy_pct | 86.425000 | 86.425000 | 84.416071 |
| nll | 0.244479 | 0.227881 | 0.187998 |
| multiclass_brier | 0.111309 | 0.107772 | 0.095387 |
| ece_10_equal_width | 0.033301 | 0.009097 | 0.010797 |
| mean_confidence_pct | 96.700828 | 94.037550 | 95.029837 |
| NOVA 2 precision | 0.161403 | 0.161403 | 0.180946 |
| NOVA 2 recall | 0.725000 | 0.725000 | 0.637500 |
| NOVA 2 f1 | 0.264027 | 0.264027 | 0.281883 |

**Development candidate grid** (all temperatures fitted by weighted NLL):

| Alpha | T | Standardized accuracy (%) | Standardized macro F1 | Standardized NOVA 2 precision | NOVA 2 recall | Constraints pass |
|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 1.287740 | 93.4869 | 0.728895 | 0.161403 | 0.7250 | Yes |
| 0.1 | 1.303574 | 93.5847 | 0.727821 | 0.161193 | 0.6750 | Yes |
| 0.2 | 1.316549 | 93.8064 | 0.734123 | 0.174775 | 0.6625 | Yes |
| 0.3 | 1.326716 | 93.9501 | 0.736114 | 0.180946 | 0.6375 | Yes |
| 0.4 | 1.334178 | 94.0804 | 0.744500 | 0.210229 | 0.6125 | No |
| 0.5 | 1.339074 | 94.2272 | 0.752086 | 0.233752 | 0.6125 | No |
| 0.6 | 1.341572 | 94.5043 | 0.767131 | 0.285279 | 0.6125 | No |
| 0.7 | 1.341875 | 94.6917 | 0.777297 | 0.331992 | 0.5875 | No |
| 0.8 | 1.340220 | 94.8194 | 0.776387 | 0.341804 | 0.5375 | No |
| 0.9 | 1.336888 | 94.8440 | 0.772235 | 0.340601 | 0.4875 | No |
| 1.0 | 1.332203 | 95.0637 | 0.782597 | 0.415868 | 0.4625 | No |

Alpha 0.3 maximized standardized macro F1 among feasible candidates. The minimum recall was 0.625 and minimum standardized accuracy 92.986884%; required macro-F1 gain for nonzero alpha was 0.005. The selected gain was 0.007219. Alpha 0.4 and above failed recall constraints. Temperature-only did not change any class counts.

**Unweighted confusion matrices (rows recorded NOVA, columns predicted NOVA):**

baseline:

| Actual / predicted | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 435 | 27 | 30 | 8 |
| 2 | 5 | 58 | 16 | 1 |
| 3 | 21 | 40 | 920 | 19 |
| 4 | 20 | 19 | 164 | 3297 |

selected:

| Actual / predicted | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 433 | 20 | 36 | 11 |
| 2 | 7 | 51 | 20 | 2 |
| 3 | 20 | 33 | 926 | 21 |
| 4 | 19 | 13 | 153 | 3315 |

The adjusted model classified 4,725 of 5,080 correctly, versus 4,710 originally (net +15). NOVA 2 true positives declined from 58 to 51, false positives from 86 to 66, and false negatives increased from 22 to 29. These are net counts, not paired transition identities. Unweighted NOVA 2 F1 was nearly unchanged (0.517857 to 0.517766); balanced accuracy fell 2.009 pp. Standardized NOVA 2 precision increased from 16.14% to 18.09% but is still low. Full prior adjustment achieved greater standardized accuracy and macro F1 but retained only 46.25% NOVA 2 recall, so was rejected by the predeclared development constraints. Calibration improved in development; ECE was slightly lower with temperature-only than combined adjustment. A fitted improvement on the same development data is not a generalization result. No claim of improvement for India or raw/OCR input is supported.

### 23.2 Reproducibility

- frozen_adjustment_sha256: `5cab4b4772f2c0c7edc93b8dc39c10cdf55a155c4c5c9df4c8806e79e20956b0`
- development_logits_sha256: `1aef85000ff9884670ec1972fdca64c9b7e06b0e25fbe335e32a660625231c4f`
- Development SHA-256: `0fc4c45c2c4bba6c3910f83da2e30ea304deaa6552c382bb06a3aaf38b626c6a`
- Holdout SHA-256: `6523cfa3fb11bc91df1d8243d4fe11e51e0f92a0cff29a0e0398f3766a6b24ea` (from I7.2, not opened in I7.3).
- Actual full metrics remain in `data/integration/evaluation_i7_3/i7_3_development_metrics.json`. Frozen combined parameters alpha=0.3, T=1.326715596542784; temperature-only alpha=0, T=1.2877402988474436.

## 24. I7.4 — Holdout protocol (planned)

Evaluate once on all 5,080 untouched structured-text holdout products. Verify frozen configuration, development logits, samples, checkpoint, train file and manifest hashes. Recheck development/holdout group, code and normalized OFF-text disjointness; use same logit extraction and metric functions. Compare exactly three settings on identical holdout logits: original, frozen temperature-only and frozen combined. No fitting or searching. Report unweighted and prior-standardized metrics, calibration, per-class metrics, confusion matrices and paired correctness counts.

**Predeclared acceptance rule, proposed by assistant before holdout results:** combined adjustment must gain at least 0.005 standardized macro F1, lose no more than 0.5 percentage points standardized accuracy, lose no more than 0.10 absolute NOVA 2 recall, and improve standardized NLL by at least 0.0001 compared with holdout original baseline. If it fails any gate, retain original class decisions and use the frozen temperature-only configuration only if it improves standardized NLL by at least 0.0001; otherwise retain original probabilities too. ECE and Brier are also reported. This is a limited acceptance test of preselected candidates, not a search for new values or statistical proof of improvement. Do not optimize on the holdout; further experimentation requires fresh evaluation data. The recommendation is restricted to screened curated structured text and not validated for India, raw ingredients or OCR. The reference distribution is not a retail prior. Only 80 NOVA 2 holdout examples; conclusions must reflect this uncertainty.

Revision 0.11 — I7.3 completed; I7.4 planned, not yet run. Previous findings preserved.


## 25. I7.4 — Frozen holdout results (completed)

**Source:** Member 1 supplied terminal JSON, 26 November 2026 local time. All 5,080 holdout examples evaluated on MPS:0, Torch 2.15.0, Transformers 5.24.0. Logit collection elapsed 65.218845 s; 20 parity checks. Development/holdout exact code/group/text overlap reported as zero. Same model hashes as I7.3; no fitting or weight change. This is a separate stratified curated holdout, not an independent external gold-standard or Indian-market sample.

### 25.1 Metrics

**Unweighted**

| Metric | Original | Temperature only | Combined |
|---|---:|---:|---:|
| accuracy_pct | 92.165354 | 92.165354 | 92.618110 |
| macro_f1 | 0.817791 | 0.817791 | 0.835562 |
| balanced_accuracy_pct | 89.807143 | 89.807143 | 89.514286 |
| nll | 0.279397 | 0.257581 | 0.219577 |
| multiclass_brier | 0.129418 | 0.124195 | 0.112671 |
| ece_10_equal_width | 0.042616 | 0.015682 | 0.016472 |
| mean_confidence_pct | 96.377207 | 93.572072 | 94.246197 |

| NOVA | Metric | Original | Temperature only | Combined |
|---|---|---:|---:|---:|
| 1 | precision | 0.908911 | 0.908911 | 0.914172 |
| 1 | recall | 0.918000 | 0.918000 | 0.916000 |
| 1 | f1 | 0.913433 | 0.913433 | 0.915085 |
| 2 | precision | 0.412121 | 0.412121 | 0.496241 |
| 2 | recall | 0.850000 | 0.850000 | 0.825000 |
| 2 | f1 | 0.555102 | 0.555102 | 0.619718 |
| 3 | precision | 0.796429 | 0.796429 | 0.792801 |
| 3 | recall | 0.892000 | 0.892000 | 0.903000 |
| 3 | f1 | 0.841509 | 0.841509 | 0.844320 |
| 4 | precision | 0.991793 | 0.991793 | 0.991231 |
| 4 | recall | 0.932286 | 0.932286 | 0.936571 |
| 4 | f1 | 0.961119 | 0.961119 | 0.963126 |

**Prior standardized**

| Metric | Original | Temperature only | Combined |
|---|---:|---:|---:|
| accuracy_pct | 92.462489 | 92.462489 | 92.972090 |
| macro_f1 | 0.726676 | 0.726676 | 0.747923 |
| balanced_accuracy_pct | 89.807143 | 89.807143 | 89.514286 |
| nll | 0.275296 | 0.251750 | 0.207586 |
| multiclass_brier | 0.125817 | 0.120542 | 0.107494 |
| ece_10_equal_width | 0.043100 | 0.017149 | 0.020101 |
| mean_confidence_pct | 96.719341 | 94.058267 | 94.961617 |

| NOVA | Metric | Original | Temperature only | Combined |
|---|---|---:|---:|---:|
| 1 | precision | 0.796816 | 0.796816 | 0.814150 |
| 1 | recall | 0.918000 | 0.918000 | 0.916000 |
| 1 | f1 | 0.853126 | 0.853126 | 0.862077 |
| 2 | precision | 0.159185 | 0.159185 | 0.211801 |
| 2 | recall | 0.850000 | 0.850000 | 0.825000 |
| 2 | f1 | 0.268151 | 0.268151 | 0.337068 |
| 3 | precision | 0.764404 | 0.764404 | 0.765063 |
| 3 | recall | 0.892000 | 0.892000 | 0.903000 |
| 3 | f1 | 0.823288 | 0.823288 | 0.828328 |
| 4 | precision | 0.993972 | 0.993972 | 0.993545 |
| 4 | recall | 0.932286 | 0.932286 | 0.936571 |
| 4 | f1 | 0.962141 | 0.962141 | 0.964217 |

### 25.2 Confusion matrices and paired results

Unweighted matrices (rows recorded OFF NOVA; columns predicted NOVA). Temperature-only matrix equals original.

**Original**

| Actual / predicted | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 459 | 21 | 20 | 0 |
| 2 | 2 | 68 | 9 | 1 |
| 3 | 26 | 56 | 892 | 26 |
| 4 | 18 | 20 | 199 | 3263 |

**Combined**

| Actual / predicted | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| 1 | 458 | 14 | 28 | 0 |
| 2 | 3 | 66 | 10 | 1 |
| 3 | 27 | 42 | 903 | 28 |
| 4 | 13 | 11 | 198 | 3278 |

Original: 4,682 correct, 398 incorrect. Combined: 4,705 correct, 375 incorrect. Both correct 4,673; original only correct 9; combined only correct 32; both wrong 366. There were 54 changed predictions, of which 13 switched between incorrect labels. Net gain 23 (+0.452756 pp unweighted). NOVA 2: 68 true positives, 97 false positives, 12 false negatives before; 66 true positives, 67 false positives, 14 false negatives after. Unweighted precision rises from 41.21% to 49.62%, recall falls from 85.00% to 82.50%. Standardized precision rises from 15.92% to 21.18%; low precision remains a material limitation. Only 80 recorded NOVA 2 examples; class-specific results are uncertain. These results do not prove improvement for each product or deployment population.

### 25.3 Predeclared acceptance gates

| Gate | Observed result | Pass |
|---|---:|---|
| Standardized macro F1 gain >= 0.005 | +0.021246 | Yes |
| Standardized accuracy loss <= 0.5 pp | +0.509601 pp gain | Yes |
| NOVA 2 recall loss <= 0.10 absolute | -0.025 | Yes |
| Standardized NLL improvement >= 0.0001 | 0.067710 | Yes |

Temperature-only also passed its NLL-improvement gate (0.023545). The frozen combined configuration is accepted under the predeclared engineering criteria for the screened curated structured-text scope. This is not proof of a universal improvement. Standardized estimates reweight the stratified cohort to the original screened curated group frequencies; they are not measured retail prevalence.

### 25.4 Reproducibility

Parameters: original alpha=0, T=1; temperature-only alpha=0, T=1.2877402988474436; combined alpha=0.3, T=1.326715596542784. Formula: `softmax((logits + alpha * log(target_prior / train_prior)) / temperature)`. Apply prior bias before temperature, use unrounded logits, NOVA order [1,2,3,4].

- holdout_sha256: `6523cfa3fb11bc91df1d8243d4fe11e51e0f92a0cff29a0e0398f3766a6b24ea`
- development_sha256: `0fc4c45c2c4bba6c3910f83da2e30ea304deaa6552c382bb06a3aaf38b626c6a`
- frozen_adjustment_sha256: `5cab4b4772f2c0c7edc93b8dc39c10cdf55a155c4c5c9df4c8806e79e20956b0`
- development_metrics_sha256: `535624ddae3fdc371650dee8521e2988c0ad5e998fd7d06f09733d92e484260c`
- predictions_sha256: `da23f8a361d2d1ae8608870e4e8f0a8a9debca5747088f1b7a58160f62946d27`
- Model config.json: `d3b802f47aed03250232fbfe367bf58e34393bc0831b5853a26a2099268aa829`
- Model model.safetensors: `d688e4f77855d2797149caf80006cfd9859b0f92424e7da6d2feb48236cc371b`
- Model tokenizer.json: `0d3aef594edd5f9b53e7f814277a9171dc70ff93eb66bda6e01f7aa53997d963`
- Model tokenizer_config.json: `e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd`

Full local evidence: `data/integration/evaluation_i7_4/i7_4_holdout_metrics.json` and `holdout_predictions.csv`. The report is based on supplied output, not independent model reruns.

## 26. I7.5–I7.6 — Fine-tuning decision and inference strategy

**Decision proposed after the predeclared holdout test:** defer fine-tuning for the current prototype. Preserve the original DistilBERT weights, tokenizer and predictor. The adjustment meets the curated holdout acceptance criteria, but is not a validated solution to India-specific error patterns or OCR failure. The holdout is now consumed: do not use it to fit additional parameters. Any new tuning requires separately prepared evaluation data and preferably independently reviewed labels.

**Recommended implementation policy:** default to original model behavior (alpha=0, T=1) for the general app, especially India-tagged products, raw OFF/USDA ingredients, OCR and manual text, or unknown provenance. Offer combined adjustment only as an explicit research/evaluation mode for screened-curated structured OFF data matching the evaluated scope; do not automatically enable it for arbitrary barcode lookups merely because structured text exists. Temperature-only also improved holdout probability quality but is not validated for other scopes; keep it as a research comparator. Do not claim India improvement or market-wide calibration. Label scores as model scores, not a guarantee of correctness. Do not interpret NOVA as a general health or safety score.

For product lookup keep the recorded OFF classification separate from the predicted NOVA and identify ingredient provenance. Preserve raw and structured text for inference; comparison-cleaned text remains for BDA matching only. Reject blank or symbol-only inputs and flag suspicious fragments; require review of OCR text, but allow legitimate single ingredients. Preserve discrepancies between OFF and USDA, and avoid presenting source agreement as accuracy.

**Status:** I7.5–I7.6 decisions documented; the policy file was subsequently listed in the local `docs/` directory. Backend implementation and testing remain pending. No new training or inference is required for the prototype at this point. I8 onward is documented in `docs/BACKEND_DEVELOPMENT_REPORT.md`, including the revised Spark/PostgreSQL/Kafka architecture. Member 3’s mobile branch remains separate.

Revision 0.12 — I7.4 completed, fine-tuning decision and scope-limited inference policy documented; previous revision statements are historical.

## 27. Research closeout and document split

The numerical findings, model parameters, experimental hashes and limitations from I1–I7 are retained. Revision 0.14 included an unexecuted SQLite catalogue design. The user explicitly confirmed that I8.3.1 had not been implemented and chose the richer Big Data stack; this old design is not an active instruction. I8.1 and I8.2 observations are carried forward in the new backend report. Retain older report revisions in Git history if needed; use revision 1.0 as the current research reference. Further DL demonstrability (e.g., carefully evaluated explanations or model comparison interface) is a deferred topic, not a completed feature or a requirement for the current backend work.
