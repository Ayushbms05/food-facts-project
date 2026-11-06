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