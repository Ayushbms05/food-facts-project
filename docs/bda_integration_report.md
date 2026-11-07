# Big Data Analytics — Complete Observations and Integration Report

**Food Label AI Project | Version 1.0 | 6 November 2026**  
**Scope:** Completed BDA work through Step 23 (Open Food Facts, USDA, ingredient comparison, source-aware curation).  
**Status:** Team-generated experimental observations, not a published research benchmark. Counts refer to the downloaded snapshots and the team's Spark programs.

## Executive summary

We processed two independently maintained food-product datasets with Apache Spark: **Open Food Facts (OFF), 4,893,632 product records in Parquet**, and **USDA FoodData Central Branded Foods, 2,010,212 records in CSV**. After identifier normalization, check-digit validation, USDA version selection, and source integration, the project produced **368,055 source-linked product records with 368,055 unique normalized GTINs**.

Of the matched products, **313,501** initially had ingredient text in both sources; **171,987** lists matched after basic text normalization. More careful cleaning (HTML markup, whitespace and punctuation) produced **71,326 additional matching pairs**, giving **243,313 comparable matching pairs** after improved cleaning, or **77.61% of 313,499 pairs retaining usable cleaned text in both sources**. **70,186** pairs remained textually different. The curated dataset flagged **124,742 records** for review, and identified **302,686 preliminary OFF ingredient + NOVA candidates** within the matched subset. None of these candidate counts constitutes an independently verified training set.

**Main BDA contribution so far:** real multi-source ingestion (nested Parquet + CSV), schema inspection, data-quality profiling, GTIN standardization and checksum verification, time-aware deduplication, cross-source joins, ingredient agreement analysis, provenance preservation, and a reusable curated Parquet output. Images, PDFs, streaming events and benchmark measurements remain planned; they are not represented as completed work.

## 1. Project goal and data-source coverage

The overall project combines Big Data Analytics, Deep Learning and a mobile application for food-label interpretation. The BDA component is responsible for collecting heterogeneous data, checking reliability, integrating records and providing traceable datasets to downstream components. For the teacher's **variety** requirement, independent sources and data types must be distinguished: reading one database twice in different formats is not the same as integrating two independent databases.

| Source / type | Formats and relevant fields | Status at Step 23 |
|---|---|---|
| Open Food Facts product database | Nested Parquet; barcode, translated names and ingredient lists, country tags, NOVA, nutrients | Ingested, profiled, India subset prepared |
| USDA FoodData Central Branded Foods | CSV; `branded_food.csv` and `food.csv`, `fdc_id`, `gtin_upc`, descriptions, ingredients, publication dates | Ingested, deduplicated by version, matched to OFF |
| Source-linked curated products | Parquet; preserved OFF and USDA texts, normalized texts, agreement flags and provenance | Created and validated |
| Product photos and OCR | Image files plus OCR/metadata JSON | Planned; not ingested |
| FSSAI reference documents | PDFs / extracted reference text | Planned; not ingested |
| Scan events from the mobile application | JSON records, eventually streamed; real and simulated clearly labeled | Planned; not ingested |
| OFF live barcode API | JSON lookup | Planned app integration |

## 2. Open Food Facts: initial exploration

**Observed file/schema:** `data/food.parquet` had **111 columns**. Relevant data types were `code: string`, `brands: string`, `countries_tags: array<string>`, `nova_group: int`, `nutriscore_grade: string`, `product_name: array<struct<lang:string,text:string>>`, `ingredients_text: array<struct<lang:string,text:string>>` and a nested `nutriments` array. The text fields are not plain strings: the analysis must select language-specific entries.

| OFF profiling metric | Observed count | Share / interpretation |
|---|---:|---|
| All product rows | 4,893,632 | Global product snapshot |
| Rows with nonempty ingredient entries / text | 1,349,598 | 27.58% of all rows |
| Rows with valid NOVA group 1–4 | 1,184,181 | 24.20% of all rows |
| Rows with both ingredient entries and NOVA | 1,129,792 | 23.09%; preliminary global candidates |
| Rows tagged as sold in India | 25,844 | 0.53% of global rows |
| India-tagged rows with ingredients and NOVA | 7,101 | 27.48% of India-tagged rows |
| Indian candidates with nonblank English text | 7,048 | 99.25% of 7,101 candidates |
| Indian candidates without English entry | 53 | Other languages or no English entry |
| Indian candidate rows with missing barcode | 0 | Within this filtered subset only |
| Barcodes repeated in Indian candidate subset | 0 | Within this filtered subset only |

**India NOVA distribution (7,101 candidates):**

| NOVA group | Count | Share of Indian candidates |
|---|---:|---:|
| 1 | 1,291 | 18.18% |
| 2 | 262 | 3.69% |
| 3 | 1,114 | 15.69% |
| 4 | 4,434 | 62.44% |

**Language-entry counts among India candidates (top ten):** `en` 14,093; `main` 7,101; `fr` 180; `zh` 35; `de` 24; `es` 22; `it` 12; `nl` 12; `hi` 8; `th` 6. These are counts of **entries** within multilingual arrays, **not unique products**; `main` is an entry tag rather than a language such as English or Hindi.

**Interpretation:** India-tagged records are a small, probably incomplete view of the Indian market. The candidate set is strongly imbalanced toward NOVA 4, and NOVA 2 has few examples. Country tags are not proof of the product's manufacture or sale location. Any India-focused model will need separate evaluation, and labels will require independent quality checks.

**Output:** `data/india_candidates.parquet/`, a filtered candidate subset, not a verified training dataset.

## 3. Practical processing issue and memory mitigation

The first full-file Spark aggregation failed with **`java.lang.OutOfMemoryError: Java heap space`**. The schema inspection completed, but scanning nested Parquet data at the original settings exceeded available Java heap. This was a real infrastructure observation, not evidence of a corrupt download.

The subsequent run succeeded using a **local single processing thread**, a **4 GB driver heap** on the 16 GB Mac, **32 MB file partitions**, **256-row Parquet reader batches**, and **selection of only needed columns**. The exact settings used included `.master("local[1]")`, `spark.driver.memory=4g`, `spark.sql.files.maxPartitionBytes=32m`, and `spark.sql.parquet.columnarReaderBatchSize=256`. Spark's schema-aware column selection reduced unnecessary data handling.

**Limitation:** No controlled before/after runtime or RAM benchmark has been recorded yet. The current result establishes a successful configuration, not a measured speedup. The team also has a Mac with 8 GB RAM, so raw full-dataset processing should remain on the better-equipped laptop where practical.

## 4. USDA FoodData Central: structure and quality

**Source snapshot:** October 2026 branded-food CSV archive. The archive listing showed **18 entries** (including the containing directory) and **3,115,750,192 bytes of uncompressed contents**. The key CSV tables observed were `branded_food.csv` (960,715,079 bytes), `food.csv` (377,990,607 bytes) and `food_nutrient.csv` (1,506,159,976 bytes, not yet integrated).

`branded_food.csv` contains **21 columns**, including `fdc_id`, `gtin_upc`, `ingredients`, `brand_name`, `modified_date` and `market_country`. `food.csv` contains **8 columns**, including `fdc_id`, `description`, `data_type` and `publication_date`. The internal join key is `fdc_id`; the cross-source product identifier is `gtin_upc`.

| USDA metric | Observed value | Interpretation |
|---|---:|---|
| Rows in `branded_food.csv` | 2,010,212 | Rows, not unique products |
| Rows in `food.csv` | 2,010,212 | Same row count; equality alone does not prove perfect 1:1 join |
| Branded rows with a nonblank barcode | 2,010,212 | 100.00% |
| Branded rows with nonblank ingredients | 2,004,844 | 99.73% |
| Branded rows with both fields | 2,004,844 | Ingredient coverage available for matching |
| Rows passing initial numeric/length GTIN filter | 1,964,629 | 97.73% of branded rows |
| Distinct normalized GTINs after initial filter | 433,505 | Much fewer than USDA rows |
| GTINs occurring more than once | 380,773 | 87.84% of distinct GTINs repeat |
| Highest observed records per GTIN | 43 | Multiple historical/version records possible |
| Selected latest, checksum-valid USDA GTINs | 431,465 | One selected row per identifier |
| Selected rows with missing publication date | 0 | In the curated USDA subset |

**Top five repeated USDA GTINs:** `00027000732304` (43 records); `00016000441071` (35); `00042800109538` (32); `00846675005540` (32); `00051000003669` (31).

**Interpretation:** Repetition can reflect product submissions or updated versions, so blindly deleting duplicates would lose version history. The pipeline joined branded and food records on `fdc_id`, parsed `publication_date`, ranked rows per normalized GTIN by the latest publication date, and used higher numeric `fdc_id` as a deterministic tie-breaker. This selects the latest *published* record; it cannot guarantee the latest store-shelf formulation.

**Output:** `data/integration/usda_latest.parquet/` with **431,465** selected identifier-level product records.

## 5. Barcode normalization, validation and linkage

The two databases use different barcode fields: `code` in OFF and `gtin_upc` in USDA. The pipeline retained digit-only values of lengths **8, 12, 13 or 14**, then left-padded them with zeros to a common **14-character GTIN representation**, retaining string type so leading zeros were not lost. Structural check digits were tested on the normalized identifiers.

| Barcode analysis | Observed value | Meaning |
|---|---:|---|
| Distinct initially acceptable OFF identifiers | 4,821,930 | Valid numeric length, not yet check-digit verified |
| Distinct initially acceptable USDA identifiers | 433,505 | Valid numeric length, not yet check-digit verified |
| Shared distinct normalized identifiers before checksum | 369,935 | Appeared in both sources |
| Shared identifiers passing check digit | 368,055 | 99.49% of precheck matches |
| Shared identifiers failing check digit | 1,880 | 0.51% of precheck matches |
| Initial USDA distinct GTIN overlap in OFF | 85.34% | 369,935 / 433,505 |
| Extra matched OFF rows sharing a normalized GTIN | 1 | Identified before choosing one OFF row per GTIN |

**Interpretation:** Matching identifiers link records, not necessarily identical formulations or publication dates. A passing check digit is a structural test, not proof that a barcode has been legitimately assigned. Leading-zero formatting and original source identifiers must be retained for auditability.

**Output:** `data/integration/shared_gtins.parquet/` was the *pre-check-digit* shared key table; the later product integration narrowed to **368,055 checksum-valid matches**.

## 6. Integrated products and first ingredient comparison

The project joined checksum-valid, curated USDA products with OFF records using `gtin14`. For the rare duplicated matched OFF identifier, one OFF row was selected, preferring an English ingredient entry. The combined dataset preserved source-specific product names, ingredients, publication date, and the OFF NOVA label.

| Source-linked result | Count | Meaning |
|---|---:|---|
| Matched unique product identifiers | 368,055 | One record per GTIN in integrated output |
| With English OFF ingredient text | 314,306 | OFF language-specific coverage |
| With USDA ingredient text | 365,741 | USDA source coverage |
| With ingredient text in both sources (initial) | 313,501 | Initially comparable pairs |
| USDA text only | 52,240 | OFF had no usable **English** ingredient entry |
| OFF text only | 805 | USDA ingredient entry absent |
| Neither had ingredient text (initial) | 1,509 | Missing from both under first check |
| Both ingredient lists + OFF NOVA label | 302,141 | Initial *both-source* NOVA candidates |

The first comparison lowercased text, trimmed whitespace and removed an introductory `INGREDIENTS:` prefix. It found **171,987 identical strings** and **141,514 different strings** among the 313,501 initially comparable pairs (an initial **54.86%** exact text agreement). Differences could reflect formatting, markup, transcription, spelling, list order, or a truly changed formulation.

**Observed examples:** Some OFF strings contained allergen HTML markup such as `<span class="allergen">WHEAT FLOUR</span>`; USDA sometimes used all caps or included an introductory label. One later pair showed `salt` versus `slat` (possible typo); another had coconut-milk ingredients described differently; another changed the order of citrus pectin/preservative. These examples show why unequal strings are not automatically different ingredient compositions.

**Output:** `data/integration/combined_products.parquet/` (368,055 records).

## 7. Improved cleaning and agreement analysis

A second comparison removed HTML tags, handled a limited set of entities, standardized punctuation, and collapsed whitespace. **This aggressive normalization is for source comparison only, not a final deep-learning tokenizer.** It can remove potentially meaningful formatting or markup; manual review remains necessary.

| Agreement category after improved cleaning | Records | Definition |
|---|---:|---|
| `same_basic` | 171,987 | Already equal under first normalization |
| `same_after_extra_cleanup` | 71,326 | Became equal after added markup/punctuation cleanup |
| `still_different` | 70,186 | Both remain populated; normalized strings differ |
| `usda_only` | 52,240 | Only USDA cleaned text remains |
| `off_only` | 805 | Only OFF cleaned text remains |
| `neither` | 1,511 | Both cleaned texts empty |
| **Total** | **368,055** | Categories are mutually exclusive and exhaustive |

**Derived observations:** Overall matching pairs after improved cleaning = **243,313** (`171,987 + 71,326`). Both-source text remaining after cleanup = **313,499** (`243,313 + 70,186`). Agreement among these cleaned comparable pairs = **77.61%**; remaining string differences = **22.39%**. The extra cleanup recovered **71,326 exact matches** from records previously classified as different.

**Reconciliation:** `neither` increased from **1,509 to 1,511**. The quality audit separately reported **2** rows where cleaning removed all originally present raw ingredient text. Hence, the initial 313,501 both-source comparison pairs become 313,499 comparable populated pairs after cleaning; this is a *change in normalized text availability*, not the addition or loss of database rows.

**Important interpretation:** A `same_basic` or `same_after_extra_cleanup` label indicates text equality under the chosen cleaning rules, not scientific evidence of identical products. `still_different` indicates remaining text disagreement, not necessarily an actual ingredient contradiction.

**Output:** `data/integration/ingredient_audit.parquet/`.

## 8. Final source-aware curated dataset (Step 23)

The pipeline joined the integrated product records with the ingredient audit by `gtin14`, kept the original OFF and USDA fields, retained cleaned versions, and generated flags: `agreement_category`, `needs_review`, `model_text_source`, `dl_candidate_off`, and `text_removed_by_cleaning`.

| Curated dataset check | Observed value | Interpretation |
|---|---:|---|
| Total curated rows | 368,055 | All matched checksum-valid identifiers |
| Unique curated GTINs | 368,055 | Exactly one record per identifier |
| Records flagged `needs_review` | 124,742 | 33.89% of curated records |
| Records flagged `dl_candidate_off` | 302,686 | 82.24% of curated matched records |
| Records flagged `text_removed_by_cleaning` | 2 | Cleaning edge cases to inspect |

**Flag definitions:** `needs_review` is true for `still_different`, `usda_only`, `off_only`, or `neither` (`70,186 + 52,240 + 805 + 1,511 = 124,742`). `dl_candidate_off` means OFF has cleaned English ingredient text and an OFF NOVA class 1–4. It is a *preliminary eligibility filter*, not confirmation of label accuracy. `model_text_source` records whether the model text originated in OFF or no OFF text was available.

**Why 302,686 is not 302,141:** The earlier **302,141** number required ingredient lists from **both** sources plus an OFF NOVA label; **302,686** requires only the cleaned OFF ingredient text and OFF NOVA label within the curated matched group. These are different filters, so their counts need not agree. Neither number should be confused with the **1,129,792 global OFF preliminary candidates** or the **7,101 Indian candidates**.

**Output:** `data/curated/source_aware_products.parquet/` - a traceable cross-source data product. The raw Parquet and USDA CSV files remain untouched.

## 9. Key interpretations, limitations and unresolved checks

- **Variety and veracity demonstrated:** Two independent publishers (OFF and USDA), two ingestion formats (nested Parquet and CSV), differing schemas, per-source identifiers, missing text and version histories. The wider variety plan (photos, OCR JSON, PDFs, event streams) is **not yet implemented**.
- **Potential coverage gain is not yet safe enrichment:** There are 52,240 matched GTINs with USDA ingredient text but no English OFF ingredient text. These are candidates for source-backed comparison or fallback, **not** permission to overwrite OFF or attach OFF NOVA labels to USDA formulations without review.
- **Product identity versus version identity:** A barcode match can connect different publication times or formulations. USDA's latest `publication_date` selects the newest published row in this extract; it is not a freshness guarantee for the real market.
- **NOVA label provenance:** NOVA group comes from OFF; no USDA-provided NOVA group was integrated. OFF labels may be derived by ingredient-based methods. Independent label review is still required before external evaluation.
- **String agreement is not ingredient equivalence:** Exact normalized text matches are useful audits; language, abbreviations, OCR mistakes, synonyms, ordering and version changes are unresolved. The presented examples were truncated in console output, not reviewed in full.
- **Regional coverage:** Only 25,844 products were tagged as sold in India; 7,101 had ingredient entries plus NOVA. The sample is too sparse and imbalanced to assume representativeness of the Indian food market.
- **Repeat counts are meaningful:** 380,773 of 433,505 initially valid-format USDA identifiers repeated; a version-aware selection is more principled than indiscriminate duplicate removal.
- **Performance is not measured yet:** The memory error and successful reconfiguration are observed. No timed Parquet-versus-CSV or Spark-versus-Pandas benchmark, peak-memory measurement, scalability curve or streaming latency result has yet been produced.
- **Reproducibility details to record:** Source snapshot dates, filenames, byte sizes, hashes, Spark/Java/Python versions, machine RAM, exact Git commit, run commands and run durations. Dataset snapshots can change over time.

## 10. Next BDA activities and handoff

**Immediate priorities:** (1) review the two text-erased rows and manually inspect a small, reproducibly sampled set of `still_different` pairs, (2) confirm consistent sample counts after cleaning and record a data dictionary, (3) record file provenance and hashes, and (4) add the current findings to Git through the `feature/bda` branch and a pull request.

**Next major variety activities:** (5) ingest a manageable OFF image/OCR metadata subset, linking images to barcode/product identifier, (6) parse selected licensed FSSAI PDF references into a traceable evidence table, and (7) later ingest real/clearly labeled simulated scan events as JSON, benchmark batch versus Spark Structured Streaming, and document privacy handling.

**Evaluation tasks:** Compare file sizes and query times under documented Spark settings; test different partition sizes and selected-column reads; measure missing data and identifier overlap by source; report both successes and exceptions. Do not claim multiple format handling is equivalent to multiple independent sources.

**DL handoff:** Offer a versioned export with `gtin14`, a source-specific ingredient text field, `nova_group`, and quality indicators. Exclude dubious joins or cross-source substituted text from the primary OFF-label training pairs until independently checked. Prevent leakage through duplicate/near-duplicate product groupings and use India-focused evaluation separately.

## 11. Reproduction map and provenance

| Step | Team script | Main output / evidence |
|---|---|---|
| OFF global quality | `src/bda/analyze_food_data.py` and `check_data_quality.py` | Global and India coverage counts |
| India extraction | `src/bda/prepare_india_data.py` | `data/india_candidates.parquet/` |
| USDA profiling | `src/bda/analyze_usda_data.py` | USDA row and field completeness |
| Cross-source GTINs | `src/bda/match_off_usda.py` | `data/integration/shared_gtins.parquet/` |
| Check digits and repeats | `src/bda/validate_matches.py` | Checksum + duplicate counts |
| Latest USDA versions | `src/bda/prepare_usda_latest.py` | `data/integration/usda_latest.parquet/` |
| Joined products | `src/bda/integrate_food_sources.py` | `data/integration/combined_products.parquet/` |
| Initial ingredient differences | `src/bda/compare_ingredients.py` | Initial same/different results |
| Improved ingredient audit | `src/bda/clean_ingredient_comparison.py` | `data/integration/ingredient_audit.parquet/` |
| Final source-aware export | `src/bda/build_source_aware_dataset.py` | `data/curated/source_aware_products.parquet/` |

**Local execution convention:** VS Code integrated terminal in project root, Python virtual environment active, Java 17 for Spark. Example: `python src/bda/build_source_aware_dataset.py`. Exact dependencies and hardware configuration should be added to the repository for repeatability. `data/` and `.venv/` are excluded from Git; only scripts, schemas, reports and safe summary outputs should be committed.

## 12. Documentation links and evidence boundaries

**Official technical references:** Open Food Facts data access: https://world.openfoodfacts.org/data ; USDA FoodData Central downloads: https://fdc.nal.usda.gov/download-datasets/ ; GS1 GTIN zero-padding guidance: https://support.gs1.org/support/solutions/articles/43000734355-what-is-the-required-format-of-gtin-in-gs1-edi-standards- ; GS1 check digits: https://www.help.gs1us.org/check-digit-calculator .

**Evidence basis:** All numeric experimental observations in this document came from the team's command outputs shared in the project conversation through 6 November 2026. Percentages were calculated from those counts; explanations and limitations are interpretations. The raw dataset snapshots and all local source files were not independently rerun in this report-generation environment. This document is a **comprehensive progress and observations report**, not a claim of completed model training, OCR ingestion, streaming analytics, or independent publication-level validation.

## Progress addendum — Steps 24–36 (7 November 2026)

**Update scope:** This addendum starts **after Step 23**. Sections 1–12 above are retained from the 6 November 2026 report, without re-running or rewriting that earlier analysis. Their descriptions of image/PDF processing as “planned” reflect the earlier reporting date; this addendum records what was subsequently completed. **Streaming, controlled scaling benchmarks and independent legal interpretation have not yet been completed.**

### 13. Step 24: Open Food Facts image and metadata collection pilot

The team selected **8 distinct products** from `data/india_candidates.parquet/` and requested product image metadata from the Open Food Facts API. Three possible image types were checked per barcode: front-of-pack, ingredients and nutrition. The pipeline created **24 JSONL metadata records**, recorded image URL, type, download status, source and retrieval time, and intentionally capped actual downloads at **3 images**.

| Image category | Checked | Available online | Downloaded in pilot |
|---|---:|---:|---:|
| Front | 8 | 8 | 2 |
| Ingredients | 8 | 6 | 0 |
| Nutrition | 8 | 7 | 1 |
| **Total** | **24** | **21** | **3** |

Availability/status totals: **18** `available_not_downloaded`, **3** `downloaded`, and **3** `not_available`. No ingredient photographs were downloaded during this deliberately capped initial pilot; this did not mean that ingredient photographs were unavailable. The **21/24 availability** figure (87.5%) applies only to this eight-product selection.

**Program / evidence:** `src/bda/collect_food_images.py` → `data/images/image_metadata.jsonl` and `data/images/`.

### 14. Step 25: Spark integration of image metadata with India products

Spark read the semi-structured image JSONL inventory, linked rows to Indian product records from Parquet by barcode, and saved an integrated inventory. It verified **24 metadata records**, **8 distinct products**, and **24/24 rows linked** to Indian products with NOVA labels. The grouped results reproduced the Step 24 availability table, with downloaded status totals 3/18/3. The successful join establishes **record linkage**, not photographic or label-text accuracy.

**Program / evidence:** `src/bda/analyze_image_metadata.py` → `data/images/image_inventory.parquet/`.

### 15. Step 26: Ingredient-image download pilot

Using the six existing ingredient-image URLs rather than repeating API lookups, the team downloaded **6/6 ingredient photographs**, with **0 download failures**. The original download logs retained barcode, image URL, local path, byte size, status and retrieval time.

**Barcodes:** `8908014814014`, `8903363004604`, `8902979052405`, `8906117091509`, `8903553002809`, `3948764032707`.

**Program / evidence:** `src/bda/download_ingredient_images.py` → `data/images/ingredients/` and `data/images/ingredient_downloads.jsonl`.

### 16. Steps 27–29: Initial OCR, enhancement and Spark quality audit

**Initial Tesseract OCR (Step 27):** All six photographs produced a nonempty text result without a reported command failure: **6 technical successes, 0 empty outputs, 0 failed OCR calls**. The program counted **154 whitespace-separated words** in total and **0.547 seconds** of measured OCR processing. These are execution counts and runtime measurements, **not accuracy scores**. The user visually judged **2 of 6 photographs** to be readable: `8902979052405` and `8903553002809`.

**Enhancement experiment (Step 28):** Images were enlarged when small, converted to grayscale, contrast-adjusted and sharpened. The 6 enhanced images were each processed with **PSM 6 and PSM 11**, producing **12 enhanced OCR attempts**. Word-count totals increased to **552** (PSM 6) and **587** (PSM 11), versus the initial 154. Measured OCR-only times were **1.284 seconds** and **1.430 seconds**, respectively, versus 0.547 seconds initially; the enhanced figures **exclude image-preprocessing time** and come from separate runs. In low-quality images, output often became longer but visibly more garbled.

| Barcode | Initial PSM 6 words | Enhanced PSM 6 words | Enhanced PSM 11 words | Readable by eye |
|---|---:|---:|---:|---|
| `8908014814014` | 1 | 181 | 185 | No |
| `8903363004604` | 4 | 68 | 88 | No |
| `8902979052405` | 43 | 38 | 31 | Yes |
| `8906117091509` | 16 | 123 | 106 | No |
| `8903553002809` | 88 | 94 | 97 | Yes |
| `3948764032707` | 2 | 48 | 80 | No |
| **Total** | **154** | **552** | **587** | **2/6 readable** |

**Spark quality audit (Step 29):** The original and enhanced JSONL results were joined with Indian product Parquet; the audit contained **6 records**, one per barcode, and all **6/6** linked to Open Food Facts **NOVA group 4**. A preview-based note marked `8902979052405` as having a recognizable ingredient list; it was not a full accuracy certification. The all-NOVA-4 pilot is a sampling limitation, not a population estimate.

**Observed OCR issues:** `8902979052405` included recognizable milk, stabilizer and cashew-list language; `8903553002809` generated significant readable/promotional packaging text about foods such as pizza and burgers rather than a clean ingredient-only block. Low-quality source images often yielded text with little apparent meaning. An OCR status of `success` or `text_detected` means nonempty text, not correct ingredients.

**Programs / evidence:** `src/bda/run_ocr_pilot.py` → `data/ocr/ocr_results.jsonl`; `src/bda/improve_ocr.py` → `data/ocr/improved/ocr_comparison.jsonl`; `src/bda/analyze_ocr_quality.py` → `data/ocr/ocr_quality_audit.parquet/`.

### 17. Steps 30–31: Manually referenced Word Error Rate (WER)

A human transcribed visible packaging text for **two photographs**. The same case-insensitive word-tokenization rule was applied to each reference and OCR result. The scripts computed word-level edit distance divided by reference length: `WER = (substitutions + deletions + insertions) / reference words`. These measurements concern **whole-image visible text**, not a separately cropped/annotated ingredient-only ground truth. Reference transcripts and source images were local, and the numbers below are team-reported program outputs.

| Product barcode | Human reference words | Original PSM 6 WER | Enhanced PSM 6 WER | Enhanced PSM 11 WER |
|---|---:|---:|---:|---:|
| `8902979052405` | 45 | **2.22%** (1 edit) | 40.00% (18 edits) | 57.78% (26 edits) |
| `8903553002809` | 119 | 58.82% (70 edits) | **48.74%** (58 edits) | 60.50% (72 edits) |
| **Pooled (164 reference words)** | **164** | **43.29%** (71 edits) | **46.34%** (76 edits) | **59.76%** (98 edits) |

**Interpretation:** The original, unenhanced PSM 6 image was markedly better for the first product. Enhanced PSM 6 performed best for the second. There is **no universal winning configuration** established by these two examples. Pooled WER is derived from summed edits / summed reference words, **not** a simple average of image-specific percentages. Image-dependent segmentation, layout and visibility influence results; a two-image study is insufficient for generalizable OCR-performance claims.

**Programs / evidence:** `src/bda/evaluate_ocr_accuracy.py`, `src/bda/evaluate_ocr_batch.py` → `data/ocr/ocr_accuracy_results.jsonl` and `data/ocr/ocr_accuracy_batch.jsonl`; manually transcribed references in `data/ocr/reference/`.

### 18. Step 32: NOVA-stratified product-image availability audit

To address the initial NOVA-4-only bias, Spark selected **8 products from each NOVA group** (32 total) from the Indian candidate subset. The Open Food Facts API was queried for ingredient-image metadata; **26/32** had an available ingredient image URL (**81.25% of this sample**) and **6** did not. **0 API errors** were reported. Product checking was balanced by NOVA group; **online image availability was not**.

| NOVA group | Products checked | Images available | Missing | API errors |
|---|---:|---:|---:|---:|
| 1 | 8 | 6 | 2 | 0 |
| 2 | 8 | 6 | 2 | 0 |
| 3 | 8 | 8 | 0 | 0 |
| 4 | 8 | 6 | 2 | 0 |
| **Total** | **32** | **26** | **6** | **0** |

**Program / evidence:** `src/bda/select_balanced_images.py` → `data/images/balanced_image_candidates.jsonl`. Availability is a pilot observation; it is not an estimate of image coverage across all Indian food labels.

### 19. Steps 33–34: Balanced higher-resolution downloads and image-quality inventory

The next controlled pilot selected two available ingredient photographs for each group. **8/8 images** downloaded, **2/2 per group**, with **0 failures**. Pixel dimensions and file sizes were captured from the actual downloaded images, then combined with the 32-product availability inventory and source NOVA labels in Spark.

| NOVA | Barcode | Image dimensions (px) | File size (bytes) | Short-side <600px? |
|---|---|---:|---:|---|
| 1 | `8904083301394` | 1338 × 3991 | 550,661 | No |
| 1 | `89080955` | 4160 × 3120 | 918,808 | No |
| 2 | `8901088000468` | 3060 × 4080 | 1,361,378 | No |
| 2 | `9556023331597` | 2934 × 1003 | 370,563 | No |
| 3 | `0760851638226` | 1042 × 1920 | 229,759 | No |
| 3 | `8906112665354` | 814 × 213 | 32,306 | **Yes** |
| 4 | `8901063367258` | 3024 × 4032 | 1,057,013 | No |
| 4 | `8901414006850` | 3456 × 4608 | 1,598,044 | No |

Spark checked **32 inventory rows**, verified **32/32 source NOVA labels matched**, and found exactly **8** rows with successful image downloads. Using the implemented short-side threshold of 600 pixels, **one image** (`8906112665354`, group 3) received a small-image warning. Separately, **7/8** images had a shortest dimension of at least 1,000 pixels. Resolution is not the same as readable text.

A manual CSV review sheet was created and completed for all eight images. The user indicated that approximately all images looked good and later formally recorded **8/8 `readable_by_eye=yes` and 8/8 `has_ingredient_list=yes`**; these are human judgments, not independently validated OCR ground truths.

**Programs / evidence:** `src/bda/download_balanced_images.py` → `data/images/balanced_full/` and `data/images/balanced_download_log.jsonl`; `src/bda/analyze_balanced_images.py` → `data/images/balanced_image_inventory.parquet/` and `data/images/balanced_readability_review.csv`.

### 20. Step 35: Balanced OCR across NOVA groups

All eight reviewed images were processed with Tesseract **PSM 6** and **PSM 11** on a shared input for each image, resized only when the longest side exceeded 3,000 pixels. **16/16 OCR runs** returned nonempty text; **0 empty results or reported execution failures**. Unlike the earlier enhancement test, no aggressive contrast or sharpening was applied. Inputs had been resized/re-encoded for the experiment, so these results should not be described as untouched-original-image OCR.

| NOVA | Images | PSM 6 average words | PSM 6 OCR time (s) | PSM 11 average words | PSM 11 OCR time (s) |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 72.0 | 0.681 | 67.5 | 0.804 |
| 2 | 2 | 77.5 | 1.443 | 98.5 | 1.367 |
| 3 | 2 | 85.0 | 0.551 | 80.0 | 0.512 |
| 4 | 2 | 168.0 | 1.468 | 162.0 | 1.057 |
| **Total** | **8** | **805 words total** | **4.143** | **816 words total** | **3.740** |

Across eight images, the measured Tesseract OCR-only average was **0.518 seconds/image** for PSM 6 and **0.468 seconds/image** for PSM 11 (rounded). These runs were not a controlled end-to-end benchmark because image preparation, storage I/O and other overheads were not included. The small samples do not establish that one NOVA group is inherently easier/harder to recognize, nor that the lower-time mode is more accurate.

**Qualitative examples:** Barcode `8906112665354` (only 814 × 213 px) nonetheless produced a clear ingredient snippet involving seeds, jaggery and nuts with **21 words** in each mode. Barcode `8901063367258` (3024 × 4032 px) generated largely incoherent OCR text despite being marked readable by eye; it yielded **29** (PSM 6) and **28** (PSM 11) words. Barcode `8901414006850` generated **307** and **296** words, respectively, but no reference transcript has validated how many were correct. Thus word count and resolution alone are unreliable proxies for ingredient extraction quality.

**Manual quality results:** 8 images reviewed; **8 readable**, **8 showing ingredient lists**, and **8 eligible for transcription** according to the member's CSV sheet. **WER has not yet been measured for these eight new photographs.**

**Program / evidence:** `src/bda/run_balanced_ocr.py` → `data/ocr/balanced/ocr_results.jsonl` and `data/ocr/balanced/ocr_audit.parquet/`.

### 21. Step 36: Independent FSSAI PDF reference corpus

Two local reference PDFs were parsed using `pypdf` and turned into a **page-level Apache Spark Parquet dataset**. This adds a **third independent content provider**: the Food Safety and Standards Authority of India (FSSAI). The documents are regulatory references, **not food-product observations**, and were not joined to product barcodes or used to certify product legality.

| Input filename | Pages processed | Pages with text | Extracted word tokens | Characters |
|---|---:|---:|---:|---:|
| `labelling_amendment_2026.pdf` | 7 | 7 | 4,278 | 19,297 |
| `labelling_compendium.pdf` | 39 | 39 | 10,288 | 65,233 |
| **Total** | **46** | **46** | **14,566** | **84,530** |

The counts reflect the script's regular-expression word tokenizer and extracted text character lengths. **0 pages without text** and **0 caught page-extraction exceptions** were reported. Keyword coverage (pages containing each term, case-insensitive; page counts can overlap): `ingredient` **24**, `allergen` **4**, `nutrition` **12**, `additive` **15**, `label` **36**.

**Important warning:** During the amendment PDF parse, `pypdf` warned that `fontTools` was not installed and is needed to fully interpret the encoding of a CFF Type1 `/MyriadPro-Regular` font. The script completed, but its zero exception count does **not** prove that every character was extracted correctly. A defensible next action is to install `fonttools`, rerun, and manually inspect selected pages for text fidelity, especially the amendment. Some scanned or complex pages may also need OCR-based inspection. The actual issuing authority, version and applicability of the locally named amendment file still require checking against its official source document.

Each output row preserves **document name, PDF SHA-256 hash, page number, text, word/character counts and extraction error field**; the script also stores a source-page URL field. These are useful provenance controls, but keyword hits are **not** evidence of substantive compliance.

**Program / evidence:** `src/bda/process_fssai_documents.py`; source files in `data/reference/fssai/`; output `data/reference/fssai/fssai_pages.parquet/`.

### 22. Consolidated new evidence and interpretation (Steps 24–36)

| Evidence area | Confirmed new result | Main limitation |
|---|---|---|
| Images / API | 8-product pilot: 24 image slots, 21 available; 3 initially downloaded | Small source-dependent pilot; availability is not quality |
| Cross-format image integration | 24/24 JSONL metadata records matched to Indian product Parquet | Record linkage alone says nothing about OCR accuracy |
| Ingredient-image pilot | 6/6 ingredient images downloaded, no failures | Four of six initially not readable by eye |
| OCR method comparison | 6 photos × 3 settings; 154 / 552 / 587 detected words | Higher word count can mean more noise |
| Manually referenced WER | Two references; original PSM6 pooled WER 43.29% | Only 2 photographs, whole-image texts, not an ingredient-only benchmark |
| NOVA-balanced image discovery | 32 checked (8 per group), 26 available, 0 API errors | Controlled small sample; online availability varies by group |
| Balanced image download and inventory | 8/8 downloaded; 32/32 NOVA matches; one short-side warning | Eight-image pilot; human assessment is subjective |
| Balanced OCR | 8 images, 16 nonempty results; PSM6=805 words/4.143s, PSM11=816 words/3.740s | No WER annotations for these eight images yet |
| FSSAI PDF reference ingestion | 46/46 pages returned text; 14,566 word tokens; keyword coverage measured | Font-encoding warning and no verified legal interpretation |

**Data-variety conclusion:** The BDA pipeline now demonstrates OFF Parquet + API JSON + image files, USDA CSV, image/OCR JSONL, FSSAI PDFs, and derived Spark Parquet. **Format variety and source variety are different:** the independent sources now include OFF, USDA and FSSAI; multiple OFF formats alone do not count as separate providers. The report does **not** claim that the 8–32-image samples by themselves constitute large-scale image processing.

**Verification boundary:** This addendum records command outputs and manual assessments shared by the team through **7 November 2026**. The report author did not access or independently rerun the large datasets, scanned images or the two FSSAI PDFs. All image quality and OCR accuracy findings should retain their small-sample qualifications.

### 23. Incremental reproduction map and next BDA tasks

| Step(s) | Script(s) | Output(s) |
|---|---|---|
| 24–25 | `collect_food_images.py`; `analyze_image_metadata.py` | `data/images/image_metadata.jsonl`; `data/images/image_inventory.parquet/` |
| 26–27 | `download_ingredient_images.py`; `run_ocr_pilot.py` | `data/images/ingredient_downloads.jsonl`; `data/ocr/ocr_results.jsonl` |
| 28–29 | `improve_ocr.py`; `analyze_ocr_quality.py` | `data/ocr/improved/ocr_comparison.jsonl`; `data/ocr/ocr_quality_audit.parquet/` |
| 30–31 | `evaluate_ocr_accuracy.py`; `evaluate_ocr_batch.py` | `data/ocr/reference/`; `data/ocr/ocr_accuracy_batch.jsonl` |
| 32–34 | `select_balanced_images.py`; `download_balanced_images.py`; `analyze_balanced_images.py` | `data/images/balanced_image_candidates.jsonl`; `balanced_download_log.jsonl`; `balanced_image_inventory.parquet/`; `balanced_readability_review.csv` under `data/images/` |
| 35 | `run_balanced_ocr.py` | `data/ocr/balanced/ocr_results.jsonl`; `data/ocr/balanced/ocr_audit.parquet/` |
| 36 | `process_fssai_documents.py` | `data/reference/fssai/fssai_pages.parquet/` |

**Next work (not completed):** (1) Install `fonttools`, re-extract and visually compare FSSAI sample pages; (2) add/manual-check ingredient-only transcriptions from the new 8-image balanced set and calculate WER with transparent image crops/regions; (3) document source attribution, URLs, collection dates and licensing for images/PDFs; (4) run controlled Spark batch/format/partition benchmarks with repeat measurements and hardware specs; (5) design privacy-safe, clearly flagged real or simulated scan events for Spark Structured Streaming. All new code belongs on `feature/bda`; large images, PDFs and raw datasets should remain excluded from Git unless explicitly licensed and needed.