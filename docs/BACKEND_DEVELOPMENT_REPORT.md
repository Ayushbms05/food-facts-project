# Food Facts — Backend Development and Big Data Architecture

**Living technical report | Revision 0.17 | 26 November 2026 | Prediction HTTP regression verified; ingredient quality pilot prepared**

Repository: `Ayushbms05/food-facts-project`  
Branch: `feature/bda-dl-integration`  
Companion research report: `docs/BDA_DL_EVALUATION_FINDINGS.md` (revision 1.0). Inference policy: `docs/NOVA_INFERENCE_POLICY.json`.

## 1. Executive summary and scope

The completed BDA–DL evaluation is documented separately. This report begins the engineering phase: make existing food data and the model accessible to Android, and collect permitted scan events for continuous Big Data analytics. The earlier PyArrow/SQLite catalogue builder was prepared and tested synthetically but, per the user's explicit statement, I8.3.1 was not implemented. That approach is superseded. Do not execute `build_i8_3_catalogue.py` as part of the revised plan. Prior experimental results are unaffected. PostgreSQL 17.12 is healthy, and the three-row Spark JDBC round trip is verified by supplied terminal output. The earlier persistence exercise is user-confirmed without separate before/after logs. The 10,000-row real-data Spark pilot passed: 9,993 active valid records, seven invalid check digits, 8,352 curated links (8,349 consistent checked fields, three missing English comparisons), and all staged counts verified in 113.538 seconds. Full-snapshot ETL has completed: 4,693,098 active valid OFF rows, 200,534 rejected rows, 47 ambiguous keys and 368,054 linked curated records, in 210.598 seconds. Full outputs were preserved as staged Parquet and subsequently loaded to PostgreSQL. The subsequent JDBC pilot verified all 9,993 OFF and 8,352 curated rows and their values in PostgreSQL; the first full-load attempt stopped safely before database writes because 17 OFF field values contained literal NUL characters. The revised loader corrected 21 literal NUL characters in 17 fields across 10 records, with no affected curated links. All three full tables were verified by row-digest multiset and SQL reconciliation, and published as `catalogue` in 400.997 seconds. The original staged Parquet remains unchanged. I8.4a completed: all four fixed cases changed from sequential scans to index scans, with median execution times of 1,226.155–1,268.360 ms before and 0.022–0.026 ms after. All projected lookup fingerprints were unchanged. The nonunique B-tree index is retained. I8.4b completed: all 11 lookup cases passed, including aliases, ambiguous and absent codes and curated-link handling. Six warm client lookup medians ranged from 0.229562 to 0.3716665 ms, with 180 measured calls on one connection. The user reports a successful push of the integration branch checkpoint. Remote commit ID is not independently verified. I8.5b server startup, health, documentation, all 11 expected product HTTP status codes and graceful shutdown are observed in the supplied Uvicorn log. I8.5a restricted-role verification and I8.5b automated HTTP client report are now supplied and passed: SELECT permitted and named write privileges denied on all three catalogue tables; health ready and 11/11 HTTP lookup cases verified. Observed FastAPI 0.165.0, Uvicorn 0.54.0, Psycopg 3.3.9, Psycopg Pool 3.3.3. The user supplied a clean Git working-tree message after the API checkpoint instructions; remote push and dependency pin contents are not independently confirmed. I8.6a model preflight passed: all four hashes matched and four example predictions ran on MPS. I8.6b completed: health and 11 product lookup cases, four baseline prediction cases, one reviewed synthetic OCR case and nine invalid-request cases passed; all four class-score vectors match the prior direct inference results, and post-prediction lookup remained unchanged. I8.7 quality pilot prepared; Android, Kafka, streaming and dashboard remain pending.

The user requested simple-English explanations before commands, a separate living backend report, and BDA implementation before considering additional DL demonstration features. We will explain what each tool does, why the project needs it, how it connects to the other tools, and what the verification output means. No new DL training or explainability feature is currently planned.

## 2. Revised architecture (design, not deployed)

### 2.1 Three connected pipelines

```text
BATCH:    OFF Parquet + curated OFF–USDA Parquet
                  | Spark / Spark SQL (ETL)
                  v
          Prepared catalogue Parquet -> JDBC -> PostgreSQL
                                                  ^
                                                  |
SERVING:  Android <--------------------------> FastAPI <--> DistilBERT
                                                  |
                                                  v
EVENTS:   Permitted scan events -> Kafka -> Spark Structured Streaming
                                              |             |
                                              v             v
                                      Historical Parquet  PostgreSQL analytics
                                                            |
                                                            v
                                                         Streamlit
```

ETL means extract, transform, load: read data, prepare it, store it where the app can use it. Spark performs the heavy preparation once rather than rereading millions of records for each scan. PostgreSQL provides indexed barcode lookup. FastAPI receives Android requests and calls the existing model when appropriate. Kafka retains small event messages; Spark streaming reads those messages in micro-batches, validates and aggregates them, and Streamlit presents measured results. The event path is separate from the interactive product lookup. Analytics availability should not be a condition of basic lookup success.

### 2.2 Tool responsibilities and academic evidence

| Tool | Job in simple English | Planned evidence |
|---|---|---|
| Apache Spark / Spark SQL | Prepare large tables in parallel | Row counts, physical plans, partition and throughput experiments |
| Parquet | Store analytical data efficiently | Source and output sizes; schema and lineage |
| PostgreSQL | Search the prepared catalogue quickly | Index plans, collision audit, lookup benchmarks |
| JDBC | Transfer Spark output into PostgreSQL | Batch size, partition count and measured load throughput |
| FastAPI | Connect Android, database and model | API schema, integration tests, request latency |
| Kafka | Receive and retain permitted scan messages | Topic, partition, offsets, producer/consumer verification |
| Spark Structured Streaming | Process arriving messages continuously | Micro-batch metrics, windows, checkpoint and replay tests |
| Streamlit | Show analytics to the team and teachers | Dashboard with actual event data; synthetic/demo data labelled |
| Docker Compose | Define and run infrastructure services reproducibly | Versioned compose configuration, service health, restart tests |

### 2.3 Local deployment and resource policy

Proposed: run PostgreSQL and later a single Kafka broker in Docker Compose, while running Spark locally with the existing compatible Java/PySpark environment. Run FastAPI and the unchanged DL model natively on macOS to preserve MPS acceleration; use a native Streamlit process. Do not launch the large batch ETL concurrently with all services on the 16 GB Apple Silicon Mac. Use bounded Spark parallelism and JVM memory, bounded JDBC connections, conservative container limits and persistent volumes. Exact versions, ARM64 image availability, Java compatibility, memory limits and connector coordinates will be chosen after preflight and checked against official documentation. If Docker is absent, assess installation or native PostgreSQL as an alternative; do not install duplicate services blindly. A single machine/broker is a local prototype, not a multi-node fault-tolerant production cluster. No Hadoop/Hive/Delta Lake/Airflow addition is currently necessary. Consider optional technologies only for a demonstrated requirement.

## 3. I8.1 — Repository inventory (observed)

User-supplied terminal output, 26 November 2026:

- `git status --short` returned no entries, indicating a clean working tree under the default status settings. Ignored local data and environments are not covered by this check.
- Active branch: `feature/bda-dl-integration`. Local branches also include `feature/bda`, `feature/mobile-app`, and `main`. The listed remote-tracking branches include `origin/member3-kotlin-xml` and `origin/feature/deep-learning`. A remote-tracking branch listing is not a fresh verification of the remote server or push status.
- Source directories present: `src/bda`, `src/deep_learning`, `src/integration`; a `mobile` directory is present at repository root.
- `docs/BDA_DL_EVALUATION_FINDINGS.md` (103,546 bytes) and `docs/NOVA_INFERENCE_POLICY.json` (2,711 bytes) were listed locally, along with the prior BDA and DL reports. File presence does not independently verify exact contents.
- `git ls-files | grep -Ei 'backend|fastapi|server|requirements|pyproject|sqlite|api'` returned only `requirements-dl.txt`. This path-name search did not identify a backend; it is not an exhaustive inspection of source contents.

I8.1 is complete. No source changes, new inference, model training, database creation, or live API tests were performed in this inventory.

## 4. I8.2 — Source inspection (observed)

### 4.1 Observed local resources

User-supplied terminal output: `df -h .` shows 460 GiB filesystem size, 177 GiB used, 256 GiB available, 41% capacity. `du -sh` shows `data/food.parquet` 7.5G, `data/curated/source_aware_products.parquet` 133M, `data/india_candidates.parquet` 1.4M (human-readable tool output; not exact byte counts). SQLite 3.54.1, PyArrow and PyTorch available. FastAPI and Uvicorn absent. The previously proposed SQLite builder required no new installation; this no longer applies to the revised PostgreSQL/Spark architecture. The actual derived database size, build time, and lookup latency are not yet measured.

### 4.2 Available source fields

- Original OFF: 111 fields. `code` is a string, `product_name` and `ingredients_text` are lists of language/text structures, `ingredients` is a string containing structured ingredient information, `lang` is source language metadata, `nova_group` is int32. Additional fields include `brands`, `countries_tags`, `allergens_tags`, `traces_tags`, `additives_tags`, `nutriscore_grade`, `nutriments`, `nutrition_data_per`, `quantity`, `serving_size`, and `obsolete`. Nutriments are nested records with name, value, 100g, serving, unit and prepared equivalents. Availability in the schema does not imply that any individual field is populated or correct.
- Curated OFF–USDA: 18 fields, including `gtin14`, `off_code`, OFF product name and English ingredients, recorded NOVA, USDA ID/description/ingredients/brand/date, OFF record count, comparison-cleaned texts, agreement category and quality flags. Preserve raw source descriptions separately; do not use comparison-cleaned text for model inference.
- India candidate subset: 5 fields (`code`, `product_name`, `ingredients_text`, `nova_group`, `countries_tags`). It is a derivative of the OFF source, not another independent product catalogue. Use the original `en:india` country tag to identify India-tagged records; do not append the subset or claim nationality or retail prevalence.

The existing `src/deep_learning/inference/predict.py` was supplied in full. `load_model()` chooses MPS/CUDA/CPU, loads local tokenizer and four-class checkpoint and sets evaluation mode. `predict_nova(ingredient_text, tokenizer, model)` strips text, rejects empty input, tokenizes at max_length 256 with fixed padding and truncation, applies softmax to logits, maps indices 0–3 to NOVA 1–4, returns predicted_nova, label, confidence and four class probabilities as rounded percentages. Reuse it when implementing the general app service; this step does not load the model. The existing validation checks only nonempty text, so backend input safeguards are still required. The I7 research adjustment remains research-only.

## 5. Catalogue design to preserve in the new ETL

The primary source remains broad OFF coverage (4,893,632 original rows), enriched by the curated OFF–USDA intersection (368,055 records). Neither count establishes the number of unique active valid barcodes. The earlier India subset is a derivative of OFF and is not appended as an independent source. Evaluation-only exclusions must not shrink the application catalogue. No new database counts have been measured.

- Keep original code strings and leading zeros. Validate supported ASCII GTIN lengths 8/12/13/14, reject invalid check digits/all-zero placeholders, and canonicalize valid identifiers to 14 digits. This is not proof of GS1 assignment or UPC-E expansion. Exclude explicitly obsolete records; retain unknown status as unknown.
- Retain normalized barcode collisions for audit, distinguish ambiguous from unique lookups, and do not silently choose a conflicting product. Store primary OFF records and curated source comparisons separately. Only enrich a unique matching record after checking original code, English ingredient text and recorded NOVA consistency. Linkage does not establish identical formulations. Retain failed links for review.
- Preserve product name and language, brand, original/raw English and training-style structured ingredient text where applicable, language and country metadata, recorded NOVA, nutrient names/values/units/bases, allergen and trace tags, additive tags, Nutri-Score and source provenance where available. Missing values remain unknown. Do not infer that a product is allergen-free or give a health/safety judgment from NOVA. Do not use BDA comparison-cleaned text for DL inference. Reuse the original predictor under the inference policy; do not enable research adjustments by default.
- Implement ETL with Spark column projection, transformations, staged Parquet outputs, row-accounting audits and JDBC staging loads. Design PostgreSQL tables, constraints and indexes before publishing a complete catalogue. Validate staging counts and link statuses before switching the active catalogue; avoid partial publication or destructive reruns. Measure actual size, timing and throughput after execution. Proposed table names/schema are not frozen until implementation.

## 6. Revised implementation roadmap

| Step | Work | Status |
|---|---|---|
| I8.1 | Repository inventory | Observed complete |
| I8.2 | Data schema and storage inspection | Observed complete |
| I8.3.1 | Java, Spark, Python, Docker and Mac preflight | Observed complete |
| I8.3.2a–b | Install/start Docker Desktop, verify engine and Compose; Spark smoke test | Completed |
| I8.3.2c–d | Start PostgreSQL service and test connection, SQL and persistence | Server and demo records verified; persistence user-confirmed |
| I8.3.3 | Spark/JDBC tiny end-to-end test | Verified complete |
| I8.3.4a | Bounded real-data catalogue pilot and quality audit | Completed; results in 7.9 |
| I8.3.4b | Full-scale Spark catalogue and quality audit | Completed |
| I8.3.5 | Load PostgreSQL and validate/publish | Completed; all three full tables verified; OFF barcode index created in I8.4a |
| I8.4 | ETL/load/index/lookup performance experiments | I8.4a and I8.4b completed; Git checkpoint recommended before API |
| I8.5 | Restricted database role, pooled FastAPI lookup and HTTP tests | Verified complete; Git checkpoint pending |
| I8.6 | Existing DistilBERT connection and prediction endpoint | Completed; 25 cases plus health and post-prediction regression |
| I8.7–I8.8 | Input safeguards and combined testing | Basic safeguards verified; additional offline triage pilot prepared |
| I9 | Integrate Member 3's Kotlin/XML app without overwriting branch | Planned |
| I10.1 | Set up Kafka, topic and connectivity | Planned |
| I10.2 | Publish permitted mobile scan events | Planned |
| I10.3 | Spark Structured Streaming validation and aggregation | Planned |
| I10.4 | Historical Parquet and PostgreSQL analytics storage | Planned |
| I10.5 | Streamlit dashboard | Planned |
| I10.6 | Throughput, lag, replay/restart and failure tests | Planned |
| I11 | End-to-end demonstration and final documentation | Planned |

## 7. I8.3.1 — Environment preflight (observed complete)

**Simple explanation:** Before connecting our tools, check which versions are already on the Mac. Spark runs on Java. Python controls Spark through PySpark. Docker will run database and messaging services in isolated environments. The two existing Python environments may have different package versions. We must choose a compatible set rather than upgrading everything. This step changes no project files, installs nothing and does not launch a service or process the food data. If a tool is missing or the Docker daemon is stopped, that is useful information rather than an instruction to reinstall it immediately.

Run in VS Code terminal, repository root, branch `feature/bda-dl-integration`, using the existing `.venv-integration` shell. Commands are read-only:

```bash
# Run from the food-facts-project root in .venv-integration
printf '\n--- Repository ---\n'
git branch --show-current
git status --short

printf '\n--- Mac and Java ---\n'
uname -m
sysctl -n hw.memsize
df -h .
java -version
/usr/libexec/java_home -V

printf '\n--- Python environments and Spark ---\n'
for py in .venv/bin/python .venv-integration/bin/python; do
  if [ -x "$py" ]; then
    echo "$py"
    "$py" --version
    "$py" -m pip show pyspark pyarrow torch | grep -E '^(Name|Version):'
  else
    printf '%s not found\n' "$py"
  fi
done
command -v spark-submit || true

printf '\n--- Container and package tools ---\n'
if command -v brew >/dev/null 2>&1; then brew --version; else echo 'Homebrew CLI not found'; fi
if command -v docker >/dev/null 2>&1; then
  docker --version
  docker compose version
  docker info --format 'Engine: {{.OSType}}/{{.Architecture}}; CPUs: {{.NCPU}}; RAM bytes: {{.MemTotal}}'
else
  echo 'Docker CLI not found'
fi
```

### 7.1 Observed results (26 November 2026)

| Check | Reported result |
|---|---|
| Active branch | `feature/bda-dl-integration` |
| Git changes | Modified `docs/BDA_DL_EVALUATION_FINDINGS.md`; untracked `docs/BACKEND_DEVELOPMENT_REPORT.md` (expected report split) |
| Mac architecture | arm64 |
| Physical memory | 17,179,869,184 bytes (16 GiB) |
| Filesystem | 460 GiB total, 177 GiB used, 256 GiB available, 41% capacity |
| Active Java | Temurin OpenJDK 17.0.21+9 |
| Other installed Java | Oracle JDK 26.0.1 arm64; default `java_home` selection printed JDK 26 |
| `.venv` | Python 3.12.16; PySpark 4.1.4; PyArrow and torch not found |
| `.venv-integration` | Python 3.12.16; PyArrow 26.0.1; torch 2.15.0; PySpark not found |
| `spark-submit` lookup | No output in current integration shell; not evidence that it is absent in `.venv` |
| Homebrew | 7.0.12 |
| Docker / Compose | Both CLI commands not found; engine not checked |

The warnings from `pip show` are expected for separate environments. Preserve `.venv` for Spark ETL and streaming and `.venv-integration` for DL and later API work. Use Java 17 explicitly for Spark rather than relying on default Java selection. Spark 4.1 documentation lists Java 17/21 and Python 3.10+; the reported environment is compatible in principle, with actual Spark execution still to be verified. Do not upgrade working dependencies without a demonstrated need. Docker CLI absence does not prove Docker.app is absent, so check the application first. Docker engine version, Compose version, macOS version and actual resource allocation remain unknown.

### 7.2 I8.3.2a — Historical infrastructure setup plan (executed; results below)

Install Docker Desktop for Apple Silicon if not already present, using the Homebrew cask `brew install --cask docker-desktop` or the official Mac installer. Launch Docker Desktop, complete its initial setup, and verify the CLI, Compose and engine. Start with a proposed 4 GiB memory limit and 2 CPUs if configurable; revisit these limits before Kafka or large ETL. This is a resource budget, not a measured requirement. Use native ARM64 images where possible. `docker run --rm hello-world` is a small engine smoke test, not a PostgreSQL or Kafka test. Do not create database containers before this verification. Check whether the Docker application is already installed before installing another copy.

Run a small Spark smoke test using explicit Java 17 and `.venv/bin/python`; create a local[2] Spark session, aggregate six synthetic integer rows into even/odd groups and verify counts 3/3. No real food data, model inference or database is involved. Spark and Docker tests are independent; keep original environments unchanged. After these checks, choose PostgreSQL image and JDBC versions, create a versioned Compose configuration, test a small SQL table and then transfer a tiny Spark DataFrame using JDBC. No database or service implementation is claimed yet.

Official references checked: https://docs.docker.com/desktop/setup/install/mac-install/ ; https://formulae.brew.sh/cask/docker-desktop ; https://docs.docker.com/desktop/settings-and-maintenance/settings/ ; https://downloads.apache.org/spark/docs/4.1.2/ .

### 7.3 I8.3.2a–b — Docker and Spark verification (completed)

Observed on 26 November 2026 from supplied terminal output:

| Check | Result |
|---|---|
| Docker CLI | 29.9.3, build 5457199 |
| Compose | v5.7.1 |
| Docker engine platform | linux/aarch64 |
| VM CPU allocation | 2 |
| VM reported memory | 4,107,341,824 bytes |
| hello-world | Successful twice; initial ARM64 image pull, then local image reuse |
| hello-world image digest | `sha256:1e4c05b897a751332ce85c7e9449da2d93514ea2dea6dbb0e2753c9e949b2bfa` |
| Java explicitly selected | Temurin 17.0.21+9 |
| Spark | 4.1.4; local[2], 1g driver memory, 2 shuffle partitions |
| Spark synthetic test | Six integer rows 1–6; even count 3, odd count 3; assertion passed |

Earlier screenshot showed proposed 2 CPU, 4 GB RAM, 1 GB swap, and a 460.38 GB disk limit before applying settings. A 128 GB disk limit was suggested; actual applied disk limit and swap have not been reverified. The engine output confirms CPU and approximate memory only. The Spark warnings (incubator module, hostname loopback fallback and native Hadoop library unavailable with Java fallback) were nonfatal for this test. No change to original data, model or Python dependencies. These tests establish basic local execution, not JDBC connectivity or concurrent load capacity.

Reproduction: `docker --version`; `docker compose version`; `docker info --format 'Engine: {{.OSType}}/{{.Architecture}}; CPUs: {{.NCPU}}; RAM bytes: {{.MemTotal}}'`; `docker run --rm hello-world`. For Spark use `export JAVA_HOME=$(/usr/libexec/java_home -v 17)` and `export PATH="$JAVA_HOME/bin:$PATH"`, then `PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python` with a local[2] session and `spark.range(1, 7).withColumn("remainder", col("id") % 2).groupBy("remainder").count()`. Assert `{0: 3, 1: 3}` and stop Spark.

### 7.4 I8.3.2c–d — PostgreSQL local service and SQL learning exercise (historical plan; user reports completion)

**Simple purpose:** Spark cleans millions of records; PostgreSQL stores the prepared catalogue so the app can look up one barcode quickly. First learn with three fictional products, not the real dataset. The demo schema is isolated from the future catalogue.

Proposed configuration: `infra/compose.yaml`, Compose project `foodfacts`, service `postgres`, official `postgres:17` image (major version selected; patch and digest to be recorded from actual pull), local development database `foodfacts`, initial administrator `food_admin`, port mapping `127.0.0.1:5432:5432`, named volume `postgres_data` mounted at `/var/lib/postgresql/data` (PostgreSQL 17 image layout). Health check `pg_isready -h 127.0.0.1 -U food_admin -d foodfacts`. Use a locally generated random password in `infra/.env` (ignored by `infra/.gitignore`); never commit or upload it, or paste resolved Compose config. Initial admin is for development/ETL setup only; later API should use a restricted role. Do not regenerate credentials on an existing initialized volume; changing the environment variable does not reset the database password. Port binding limits direct host exposure to loopback; no public database service is intended. Do not use `down -v` because it removes the persistent volume.

Commands from `infra/`: `docker compose config --quiet`, `docker compose up -d --wait postgres`, `docker compose ps`, `docker compose images`. Use `docker compose exec -T postgres sh -c 'PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -c "SELECT version(), current_database(), current_user;"'` to test a password-authenticated TCP session inside the container. Host-to-container JDBC connection is a separate test in I8.3.3.

SQL learning exercise: create schema `demo` and table `demo.products` with string `gtin14` primary key, name, ingredients and NOVA integer constraint 1–4. Use three fictional values (`00000000000017`, `00000000000024`, `00000000000031`), query the biscuit row and group by NOVA. The primary key automatically creates a unique B-tree index; tiny data may still use sequential scans and is not a valid performance benchmark. Store barcode as text to retain leading zeros. Use `ON CONFLICT (gtin14) DO NOTHING` for rerunnable demo inserts. Confirm persistence by `docker compose down` followed by `docker compose up -d --wait postgres` and checking the demo rows remain (no `-v`). No real food ingestion, JDBC, model inference or Kafka in this step. Exact database version, image digest, query results and persistence test remain pending.

References checked: https://hub.docker.com/_/postgres ; https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/ ; https://docs.docker.com/guides/postgresql/ .

### 7.5 I8.3.2c–d — Status update (26 November 2026)

The user replied "done move to next step" following the PostgreSQL setup, SQL and container recreation exercise. Record these as user-reported complete, not independently verified observations. Exact server patch version, image digest, health status, SQL query output and persistence evidence were not supplied. Do not invent these values. The next JDBC test also verifies that the three expected demo records are present. Capture nonsecret server version and image metadata with the next run. No real product data has been imported.

### 7.6 I8.3.3 — Small Spark JDBC ETL round trip (planned)

Purpose in simple terms: teach Spark to talk to PostgreSQL before loading millions of rows. JDBC (Java Database Connectivity) lets the JVM-based Spark engine read and write database tables using a PostgreSQL driver. The selected pgJDBC version is 42.7.14 (official 7 October 2026 release, Java 8+); pin it using `spark.jars.packages=org.postgresql:postgresql:42.7.14`. Spark downloads the driver and its dependencies on the first run if not cached. No new Python dependency is needed. Java 17 and `.venv` remain the Spark environment.

Script: `src/integration/test_i8_3_jdbc.py`. It reads the generated 48-character lowercase hexadecimal password from `infra/.env` without printing it, connects to `jdbc:postgresql://127.0.0.1:5432/foodfacts`, reads `demo.products`, verifies the three expected fictional rows, adds an uppercase name column using Spark SQL functions, writes a new `demo.spark_products` table using one output partition and illustrative JDBC batch size 2, reads back and compares every value, and confirms the original table remains unchanged. Uses `errorifexists` rather than overwriting existing tables; if a target exists or a previous write was partial, inspect it before deciding whether to remove only that demo table. Spark-created target is an instructional table, not the final catalogue schema; no automatic primary key is assumed. No large dataset processing, DL inference or performance benchmark occurs. The demo uses the existing local administrator; later API serving will use restricted credentials. No password or resolved Compose config should be shared.

Run from repository root:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.test_i8_3_jdbc
```

Expected success: 3/3 rows with all values verified and source unchanged. Verify target independently with `docker compose -f infra/compose.yaml exec -T postgres psql -U food_admin -d foodfacts -c 'SELECT * FROM demo.spark_products ORDER BY gtin14;'`. Record actual server version and image ID with the supplied check commands. The Python script passed syntax compilation only; the live driver download, JDBC connection and Spark writes are pending local execution. No actual transfer time or throughput is claimed. Next: design production catalogue staging and validate a small slice of the real Parquet data before full-scale ETL.

Official references: https://jdbc.postgresql.org/download/ ; https://jdbc.postgresql.org/changelogs/2026-10-07-42.7.14-release/ ; https://spark.apache.org/docs/4.1.4/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.jdbc.html ; https://spark.apache.org/docs/4.1.0/sql-data-sources-jdbc.html .

### 7.7 I8.3.3 — Observed JDBC round-trip results (26 November 2026)

User supplied successful terminal output from all three requested checks. PostgreSQL service `foodfacts-postgres-1` was healthy, bound to `127.0.0.1:5432`, using image `postgres:17` on `linux/arm64/v8`, image ID `87c8b42f423b`, displayed size 690 MB. Server version PostgreSQL 17.12 (Debian 17.12-1.pgdg13+1), aarch64. Image digest not supplied. The earlier container-recreation persistence exercise remains user-confirmed rather than separately logged.

Spark 4.1.4 successfully resolved and downloaded pgJDBC 42.7.14 and checker-qual 3.55.1 through Ivy. All three fictional source records were extracted, transformed with uppercase product name, written to the separate `demo.spark_products` table, read back and verified against the intended values. Original table unchanged. Independent `psql` query returned the same three records: Milk/MILK (NOVA 1), Cooking oil/COOKING OIL (NOVA 2), Biscuits/BISCUITS (NOVA 4). Host-to-container JDBC authentication and round trip verified. Incubator, hostname and native Hadoop warnings did not block completion. No measured ETL throughput or real product import yet.

### 7.8 I8.3.4a — Real-data catalogue pilot (original protocol; executed in 7.9)

Purpose: test the transformation and staging design on 10,000 actual OFF rows before the full 4,893,632-row run. The unordered Spark `limit(10000)` slice is not random or representative. Select only 17 required source fields; use local[2], 3g driver heap, 4 shuffle partitions, 32 MiB input partitions and 256-row Parquet batches, with disk-only persistence of intermediate frames. The original files remain read-only. No database or model changes. Curated Parquet is read to link pilot keys; this is not a globally deduplicated catalogue.

Script `src/integration/prepare_i8_3_catalogue_pilot.py` uses Spark SQL/DataFrame expressions rather than collecting millions of rows to Python. First-failure exclusion order: explicitly obsolete, missing/blank code, unsupported ASCII digit length (8/12/13/14), all-zero code, invalid check digit. Leading-zero canonicalization uses 14-character strings. Retain all active valid rows; count repeated keys separately. A unique key in a pilot is not a globally unique key. The checksum does not verify actual GS1 assignment or UPC-E interpretation.

Projected fields: original code, normalized key, localized name and ingredient text (English first, then declared language, main, first nonblank), explicit English ingredient field, original structured ingredient JSON, source NOVA and validated recorded NOVA 1–4 or null, brands, language, countries and India tag, allergen/trace/additive tags, nutrition JSON, Nutri-Score, quantities, obsolete status. Nested arrays are serialized as JSON text retaining nulls where applicable; missing values remain unknown. Structured ingredient JSON is preserved and exact DL text extraction is deferred to the native inference environment (existing extraction function); no comparison-cleaned text is fed to the model. This avoids installing PyArrow in the Spark environment solely for the DL module import.

Curated rows are kept separate and linked by GTIN only for keys found in the pilot, with statuses for ambiguous sample key, code mismatch, NOVA mismatch, missing English comparison, English text mismatch, or consistency under the limited comparison. English comparison uses first nonblank explicitly English text and trimmed exact string equality; conservative mismatch may arise from multiple English entries. A missing text never counts as positive agreement. No automatic USDA substitution or label overwrite. Null NOVA comparison is null-safe, not a proof of label correctness. Missing India tags are not proof of absence from India.

Provisional staging design (final PostgreSQL schema to be reviewed after pilot):

| Staged table | Grain and role |
|---|---|
| off_products | One row per active valid source record; duplicate GTINs retained; future surrogate key + nonunique GTIN index |
| barcode_conflicts | One row per repeated GTIN in pilot; future unique key and record count |
| curated_links | One row per matching curated GTIN in pilot; source provenance and link status; not automatically trusted |
| rejected_rows | One row per excluded OFF record; code, obsolete and first-failure status |

All files go under `data/backend/catalogue_pilot/`, including `i8_3_4a_pilot_report.json`. Refuse existing output directory; inspect/archive partial outputs before rerunning. Counts must reconcile: input = valid + rejected; sum of per-key counts = valid rows; distinct GTIN <= valid; curated keys unique and matched count <= distinct keys; each Parquet output read-back count equals expected count. Record field coverage, collision counts, link statuses and wall time; no extrapolation from this slice to full dataset. Full-scale ETL and constrained PostgreSQL staging/publish follow only after reviewing pilot outputs.

Commands (repo root):

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.prepare_i8_3_catalogue_pilot --self-test
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.prepare_i8_3_catalogue_pilot
```

Script syntax compilation and independent check-digit fixture arithmetic passed in report-author environment. PySpark is unavailable there; embedded Spark synthetic tests and actual Parquet processing remain pending on Member 1's Mac. No claim of measured real catalogue counts or full Spark compatibility until execution.

### 7.9 I8.3.4a — Observed pilot results (26 November 2026)

The embedded Spark self-test passed separately and again during the actual pilot. Spark 4.1.4 completed the 10,000-row unordered slice and all four Parquet count checks. Elapsed pipeline time 113.5381548749865 s (excludes session startup and synthetic self-test). No PostgreSQL writes, model inference or source changes.

| Measurement | Result |
|---|---:|
| Input rows | 10,000 |
| Active valid rows | 9,993 |
| Invalid check digit | 7 |
| Distinct accepted GTIN-14 | 9,993 |
| Collision keys / extra records | 0 / 0 |
| Nonblank product name | 9,936 |
| Nonblank raw ingredients | 9,675 |
| Nonblank explicit English ingredients | 9,534 |
| Nonblank structured ingredient JSON | 9,675 |
| Valid recorded NOVA | 9,378 |
| India tag true / unknown | 1 / 0 |
| Linked curated keys | 8,352 |
| Consistent checked OFF fields within pilot | 8,349 |
| Missing English comparison | 3 |
| Accepted keys without curated link | 1,641 |

All first-failure statuses and link categories reconcile. The accepted rows include 615 without valid NOVA labels; missing data are not fabricated. Nonblank structured JSON does not prove a nonempty parsed ingredient sequence. No duplicate keys in this slice does not establish global uniqueness. The high link rate and low India count reflect this particular unordered slice; do not extrapolate its distribution, quality or runtime to the full snapshot. Link consistency compares selected OFF fields, not equivalence between OFF and USDA ingredients. The observed Spark native-library and hostname warnings were nonfatal.

| Output under `data/backend/catalogue_pilot/` | Verified rows | Displayed disk use |
|---|---:|---:|
| off_products/ | 9,993 | 13M |
| barcode_conflicts/ | 0 | 12K |
| curated_links/ | 8,352 | 2.0M |
| rejected_rows/ | 7 | 12K |
| i8_3_4a_pilot_report.json | — | 4.0K allocated (1.2K listed size) |

### 7.10 I8.3.4b — Full-snapshot ETL (original protocol; results below)

Plain-English objective: repeat the successful small data-preparation experiment over every OFF record so the application can search all eligible barcodes, not only OFF–USDA matches. Preserve all active valid records, even if missing ingredients/NOVA or appearing in training. Duplicate canonical identifiers are quarantined logically through the conflict table; the later API must not silently pick one. Preserve both source descriptions and recorded NOVA separately. Do not start loading PostgreSQL yet.

Script: `src/integration/prepare_i8_3_catalogue_full.py` imports the verified pilot helpers for language selection, projection, grouping and source links. It adds a stricter ASCII digit/length check and null-safe checksum rejection (avoids a possible Java regex end-anchor edge case with final newlines), additional synthetic fixtures including source conflicts, and full-snapshot reconciliation. This defensive change is not evidence that the pilot misclassified any row. Input expectations are 4,893,632 OFF rows and 368,055 unique curated keys; count mismatch stops processing rather than silently accepting a different snapshot. These count checks do not establish full byte-level source identity. No category fields are added beyond the pilot projection; future category analytics will require enrichment.

Resource settings: Java 17, original `.venv`, `local[2]`, 4g driver heap, 64 shuffle partitions, 32 MiB input partitions, 256-row Parquet reader batches, adaptive execution enabled, automatic broadcast joins disabled (including adaptive broadcast). Disk-only persistence for the classified input and small-column key counts, release classified cache after writing products/rejects, narrow projections for linking. These are conservative starting choices, not benchmark-optimal settings. At least 40 GiB free on project volume is a guard, not a guarantee of adequate space (Spark temporary files may use another path). Avoid running model or other heavy programs concurrently. PostgreSQL can remain idle. The real run is not timed by extrapolating the pilot.

Output under `data/backend/catalogue_full/`: `off_products/` (all active valid rows, duplicates retained), `barcode_conflicts/` (one record per repeated key, record_count), `curated_links/` (matched curated keys, checked-field link status and record_count), `rejected_rows/` (excluded source code, obsolete and first-failure reason), `unmatched_curated/` (complete curated audit records not linked to active valid keys), `i8_3_4b_full_report.json` (counts, coverage, timing and bytes). Full link statuses rename pilot ambiguity and consistency to `ambiguous_barcode` and `consistent_checked_fields`; other mismatch statuses retained. Verify input = active + rejected, output rejected statuses, global key group sum = active, null key count zero, conflict count, distinct curated keys, matched + unmatched = curated count, and output read-back counts. All staged tables are written in `.inprogress` sibling directory and only renamed to final directory after checks and report succeed. Refuse overwrite; inspect partial outputs if execution fails. No PostgreSQL writes, inference or source changes. Future JDBC loading must additionally handle PostgreSQL text compatibility (e.g. embedded NUL), constraints and verified safe publication.

Run from repo root:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.prepare_i8_3_catalogue_full --self-test
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.prepare_i8_3_catalogue_full
```

At protocol preparation, script syntax compilation passed; PySpark execution remained pending. The subsequent local run passed, as documented below. Do not present this as a multi-node deployment.

### 7.11 I8.3.4b — Observed full-scale results (26 November 2026)

Both synthetic checks passed. The full snapshot ran on Spark 4.1.4 with local[2], 4g driver memory, 64 shuffle partitions, 32m maximum input partition, 256-row Parquet reader batches and automatic broadcast threshold -1. All counts reconciled, and staged Parquet was published. No PostgreSQL writes or inference occurred. Curated records are the previously source-linked OFF–USDA records from Step 23, including original source fields and comparison flags, not independently verified truth. The staging layer saves the transformed data on disk so JDBC loading can be retried separately.

| First-failure status | Rows |
|---|---:|
| Active valid | 4,693,098 |
| Invalid check digit | 128,889 |
| Invalid format | 71,642 |
| Missing code | 1 |
| All zero | 2 |
| Total | 4,893,632 |

Accepted 95.9021% (rounded); rejected 200,534. No obsolete first-failure records reported. Rejection means unsupported or structurally invalid identifier under the catalogue rules, not proof of a fraudulent product. Distinct GTINs: 4,693,051, of which 4,693,004 singleton keys and 47 collision keys containing 94 rows (47 extra records). All active valid records are preserved; do not make GTIN alone the primary key of the row-level table. A separate key table or surrogate record identifier with nonunique barcode index is needed; ambiguous lookup must not return an arbitrary record.

| Field coverage among accepted rows | Count |
|---|---:|
| Product name | 4,407,490 |
| Raw ingredients | 1,280,502 |
| English ingredients | 666,956 |
| Structured JSON nonempty | 1,280,573 |
| Recorded NOVA | 1,156,744 |
| India-tagged true | 25,070 |
| India tag unknown | 8,825 |

Nonempty JSON is not proven usable ingredient text. India tags are not proof of origin or market coverage. Curated input 368,055; linked 368,054, including 314,305 consistent checked OFF fields and 53,749 missing English comparisons; one unmatched preserved. This checks selected OFF fields, not equality of USDA and OFF formulations. Exact unmatched reason not established by summary.

| Staged output under data/backend/catalogue_full/ | Verified rows | Logical file bytes reported |
|---|---:|---:|
| off_products | 4,693,098 | 2,640,401,455 |
| barcode_conflicts | 47 | 29,928 |
| curated_links | 368,054 | 106,748,377 |
| rejected_rows | 200,534 | 2,448,758 |
| unmatched_curated | 1 | 7,896 |

Report: `i8_3_4b_full_report.json` (2.4K listed). Shell `du` showed 2.5G, 292K, 102M, 3.7M and 24K respectively (allocated disk usage differs from summed file bytes). Stage times: extract/transform/write 179.878535 s, coverage/conflicts 20.170223 s, source links 10.541324 s; total 210.598321 s, approximately 23,237 input rows/s for this local run. Not a JDBC throughput, controlled scalability study, or distributed cluster benchmark. No source changes, inference or real-data database writes. User-supplied terminal evidence, not independently rerun here.

### 7.12 I8.3.5a — Real-data JDBC pilot (verified)

Load the previously verified 9,993 OFF pilot rows and 8,352 curated pilot links into a new `catalogue_pilot` schema, not the final serving schema. Use the existing Java 17, Spark .venv, PostgreSQL 17.12 and pgJDBC 42.7.14. Script `src/integration/load_i8_3_catalogue_pilot.py` reads local ignored credentials without printing them, requires empty target schema, checks expected counts and unique pilot keys, validates supported scalar types, audits literal NUL characters before any write (PostgreSQL text cannot contain literal NUL), and checks all curated keys exist in OFF pilot. No automatic text cleanup. Spark writes with maximum two partitions and JDBC batch size 1000 in error-if-exists mode. Reads back both tables and verifies schema type, count and multiset equality with exceptAll in both directions. Records write times and results in `data/backend/i8_3_5a_jdbc_pilot_report.json`. The pilot intentionally uses Spark-created tables without production indexes/constraints, and no rejected/conflict audit table loading is attempted. Retain these audit Parquet files. Full load requires a full text-compatibility audit, explicit production schema, safe staging/publication, collision-aware lookup and independent reconciliation. Failed writes may leave partial pilot tables; inspect before retry, do not delete existing database/volume. Syntax compiled at preparation; subsequent local results follow.


### 7.13 I8.3.5a — Observed pilot database load (26 November 2026)

Spark 4.1.4 with pgJDBC 42.7.14 successfully loaded two real-data tables under `catalogue_pilot`. The JDBC dependency was already cached. All fields had scalar String/Integer/Long/Boolean types. No literal NUL was found in either table. Batch size 1,000, maximum two write partitions. Both source and database row counts and the complete row multiset were compared using `exceptAll` in both directions. Reported results:

| Table | Rows verified | Missing | Extra | Write time (s) |
|---|---:|---:|---:|---:|
| catalogue_pilot.off_products | 9,993 | 0 | 0 | 0.869987 |
| catalogue_pilot.curated_links | 8,352 | 0 | 0 | 0.156798 |

Write times exclude verification and are not controlled performance measurements. `SELECT COUNT(*)` independently returned 9,993. SQL displayed real products including Chamomile Herbal Tea and Lagg's peppermint herbal tea; two of the five displayed rows had NULL recorded NOVA. The final source-disagreement JOIN was entered, but its output was not included; no claim about its returned rows. The program itself verified all curated rows. Report: `data/backend/i8_3_5a_jdbc_pilot_report.json`. Full dataset not yet loaded. Pilot tables have no final production indexes or constraints.

### 7.14 I8.3.5b — Full catalogue load (implementation prepared; execution pending)

Reuse `data/backend/catalogue_full/` staged Parquet without repeating the original ETL. Script `src/integration/load_i8_3_catalogue_full.py` runs with Java 17, BDA `.venv`, Spark local[2], 4g driver memory, 64 shuffle partitions, 32m file partition target, 256-row Parquet reader batches, automatic broadcast disabled, pgJDBC 42.7.14. Check host and database filesystem free space (30 GiB initial threshold, not an estimate or guarantee of required capacity), successful JDBC connection, source report counts, scalar field types, valid 14-digit key shape, and literal NULs across each string column in all three target tables. No silent text modification. Save preflight `data/backend/i8_3_5b_preflight.json`; stop before writes if incompatible text found. `--preflight-only` audits without writes; ordinary execution performs preflight and loads only if passed.

Use `catalogue_loading` as an isolated temporary schema. Refuse if either `catalogue` or `catalogue_loading` exists; no automatic deletion or overwrite. Explicit PostgreSQL scalar types and barcode shape constraint. Main table `off_products` retains all 4,693,098 accepted records; generated BIGINT `record_id` is the primary key (local IDs may change on rebuild), not GTIN. `curated_links` targets 368,054 rows; `barcode_conflicts` targets 47 rows, each with record_count > 1. Both use GTIN primary keys. Rejected rows (200,534) and unmatched curated (1) stay in the existing analytical Parquet audit outputs rather than application database. No foreign key from curated GTIN to nonunique OFF GTIN; check linkage counts explicitly. No OFF barcode index until the planned I8.4 benchmark so its effect can be measured. These are engineering choices, not observed results.

Spark JDBC appends into explicitly created empty tables using up to two partitions and batch size 1,000. Independently count rows in SQL and add primary keys. Verify data via a signed multiset of SHA-256 digests over same ordered scalar fields serialized by Spark with explicit nulls, excluding generated record IDs; report positive/negative differences. This is strong probabilistic content verification rather than mathematical exact equality; unlike pilot exact `exceptAll`, only compact digests are shuffled. Use JDBC fetchsize 1,000; partition main readback by record_id. SQL reconciles distinct GTINs (target 4,693,051), collision keys (47), rows on collisions (94), exact conflict group counts and curated record-count links. Only after checks pass, analyze tables and rename `catalogue_loading` to `catalogue`. Preserve demo and pilot schemas. Failed loading may leave partial isolated staging schema; inspect before retry. Never delete Docker volume. Final report `data/backend/i8_3_5b_full_jdbc_report.json`. No model inference or dataset modification. Python syntax compiled; Spark/PostgreSQL integration and memory/disk behavior not tested here. No full load claim yet.


### 7.15 I8.3.5b — Full load preflight stopped safely (26 November 2026)

The first full JDBC loading attempt reached the preflight, using Spark 4.1.4 and the cached pgJDBC 42.7.14 dependency. Free storage reported: 249.77 GiB on the host and 117.26 GiB on the PostgreSQL filesystem, above the configured 30 GiB guard. The credentialed JDBC connection and row-count/key-shape checks passed (as inferred from the script reaching the NUL-specific stop after all three table checks). Observed results:

| Table / field | Rows containing literal NUL (U+0000) |
|---|---:|
| off_products.product_name | 3 |
| off_products.brands | 2 |
| off_products.ingredients_raw | 7 |
| off_products.ingredients_en | 5 |
| curated_links (all string columns) | 0 |
| barcode_conflicts (all string columns) | 0 |

Full counts rechecked: OFF 4,693,098; curated links 368,054; barcode conflicts 47. Total affected field values = 17; the number of affected rows, distinct keys, individual NUL characters and affected curated links is not established by the supplied summary. PostgreSQL character types cannot store U+0000 (official documentation: https://www.postgresql.org/docs/17/datatype-character.html). This differs from SQL NULL, an absent value. The exception `Literal NUL detected; stopped before database writes. Keep source unchanged and review the audit.` was deliberately raised after writing `data/backend/i8_3_5b_preflight.json` and before creation of the loading schema. No full-catalogue database writes in this execution. Source and staged Parquet remain unchanged. Existing pilot data is unaffected. Native Hadoop / hostname warnings did not prevent the preflight. Actual database load performance has not been measured.

### 7.16 I8.3.5b — Audited text compatibility correction (prepared; not yet executed)

Revised script: `src/integration/load_i8_3_catalogue_full_v2.py`. Preserve the original script and original preflight report. The chosen serving-only policy replaces literal U+0000 with a single ordinary space (U+0020) per occurrence, rather than joining adjacent words or dropping whole records. No change to source or staged Parquet, barcodes, recorded NOVA, JSON ingredient structures, curated records or other columns. This is a compatibility correction, not a verified restoration of the intended label text. Do not conflate it with the earlier BDA comparison cleaning or change the trained model. The correction may introduce leading/trailing spaces or empty-looking text; later inference input validation remains necessary. Curated linkage flags retain the original staged OFF comparison semantics; the audit identifies affected curated keys, which should be reviewed before treating those links as source-consistent for serving.

An explicit `--repair-nul` opt-in permits only the exact observed counts in the four named OFF text columns; any different nonzero column/count or missing expected issue stops for review. A synthetic Spark test covers embedded/repeated NUL, unchanged normal and accented text, SQL NULL, and escaped literal backslash-u sequences. Audit the affected rows using a bounded collect (maximum 17 rows, overflow guard); compare Spark replacement with Python string replacement for every collected value. Record each changed field's original and corrected text, GTIN, original code and character count in `data/backend/i8_3_5b_nul_corrections.json`, preserving duplicates in the audit. JSON escapes the NUL. Report affected row and distinct key counts, number of field changes, individual characters replaced and affected curated keys. Refuse to overwrite a differing existing correction audit. Keep audit local under ignored data; no credentials included. Reaudit every string column and OFF row count after applying the Spark projection; all NUL counts must be zero. Other tables remain unchanged. The corrected projection is evaluated lazily at load time; there is no second complete Parquet copy. The later row digest verification compares PostgreSQL against the corrected serving projection rather than unmodified original text. The prior global SQL reconciliation and isolated schema publication remain unchanged. Save the new preflight to `data/backend/i8_3_5b_preflight_v2.json` and the final successful report to the original planned path `data/backend/i8_3_5b_full_jdbc_report.json`.

Run from the project root with Java 17 and original BDA environment:

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:$PATH"
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.load_i8_3_catalogue_full_v2 --repair-nul --preflight-only
# Only if self-test and corrected preflight pass:
PYSPARK_PYTHON="$PWD/.venv/bin/python" .venv/bin/python -m src.integration.load_i8_3_catalogue_full_v2 --repair-nul
```

The first command writes only local audit reports, no database changes; the second repeats checks before the actual database load. No automatic schema deletion or retry after partial loads. Python syntax compilation passed; PySpark and the user's database are unavailable in the report-generation environment, so Spark self-test, actual audit counts and full load are unverified. User must supply results before this step can be marked complete.

### 7.17 I8.3.5b — Full PostgreSQL catalogue published (26 November 2026)

Both revised runs succeeded on the user's Mac. The synthetic replacement test passed. The `--repair-nul --preflight-only` run stopped normally without writes; the second run repeated the checks and published the full catalogue. Free host storage at preflight: 249.75 GiB; PostgreSQL filesystem: 117.26 GiB. Spark 4.1.4, JDBC 42.7.14, batch size 1,000, at most two write partitions. The correction audit reported 10 affected rows and 10 distinct GTINs, 17 changed field values (name 3, brands 2, raw ingredients 7, English ingredients 5), 21 literal NUL characters replaced, zero affected curated keys and zero remaining literal NUL counts. No source/staged Parquet modifications; no reclassification or label substitution. The audit SHA-256 was identical across both runs: `34b9c45f4aefcd1cb7b7c6efedbb2cc6602e53720bb60b96749299f60f4554b4`. Local audit `data/backend/i8_3_5b_nul_corrections.json`, revised preflight `data/backend/i8_3_5b_preflight_v2.json`, final report `data/backend/i8_3_5b_full_jdbc_report.json` (paths from the implemented script; file contents not independently retrieved here).

| Table | Rows verified | Missing/extra digests | Write seconds | Verify seconds | Primary key |
|---|---:|---|---:|---:|---|
| off_products | 4,693,098 | 0 / 0 | 123.190667 | 177.889700 | record_id |
| curated_links | 368,054 | 0 / 0 | 3.815517 | 4.394977 | gtin14 |
| barcode_conflicts | 47 | 0 / 0 | 0.076219 | 0.129708 | gtin14 |

Preflight 70.967995 s; reported overall elapsed 400.997365 s (about 6 min 41 s; measured after Spark startup). Main write rate about 38,096 rows/s, not API throughput. SQL reconciliation: 4,693,051 distinct GTINs, 47 collision keys, 94 rows on collision keys, zero conflict-count mismatches and zero curated-link count mismatches. The isolated `catalogue_loading` schema was renamed `catalogue`. No need to repeat ETL or loading. SHA-256 multiset equality is strong probabilistic evidence against the corrected projection, not byte identity with original Parquet. IDs are local and may change on rebuild; rejected and unmatched records remain in Parquet. No barcode index yet on OFF GTIN, no production or concurrent-load performance claim. Actual PostgreSQL storage sizes and manual query results are not included in the supplied log.

### 7.18 I8.4a — B-tree barcode lookup benchmark (prepared)

Objective: measure the impact of a nonunique barcode index without losing ambiguous rows. PostgreSQL's B-tree supports equality searches (https://www.postgresql.org/docs/17/indexes-types.html); EXPLAIN ANALYZE runs the query and reports measured execution statistics (https://www.postgresql.org/docs/17/sql-explain.html). Use the real `catalogue.off_products` table. Script `src/integration/benchmark_i8_4_barcode_index.py` uses standard Python and existing Docker/psql, not Spark or new dependencies. Keep other workloads stopped. Before phase confirms only the record_id unique index, verifies count 4,693,098 and four fixed fixtures: two observed known barcodes (one with recorded NOVA and one without), one actual conflict key, and a generated checksum-valid locally absent key verified by lookup. Check expected cardinalities 1, 1, 2, 0, plus fingerprints of ordered projected rows. The SELECT returns all matching records (record_id, gtin14, name, recorded_nova, English ingredients), no LIMIT hiding duplicates. Each phase has one unmeasured warmup and five measured trials per fixture, seeded shuffled order; EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON) collects server planning/execution time, scan nodes, buffer hits/reads, raw plans, median/min/max. No forced planner settings. Save baseline before creating index, then check identical fixture fingerprints and metadata before DDL. Create nonunique `idx_off_products_gtin14` on gtin14, record creation wall time and index bytes, measure same queries, require indexed plans and unchanged projected results. Keep index for backend. No automatic drops or destructive recovery. Run `--before` then `--after`; if failure occurs after creation, inspect existing index and saved evidence before retry. Output `data/backend/i8_4a_index/{before.json,index_created.json,after.json}`; report real results after user executes.

```bash
.venv-integration/bin/python -m src.integration.benchmark_i8_4_barcode_index --before
# Only after baseline success:
.venv-integration/bin/python -m src.integration.benchmark_i8_4_barcode_index --after
```

Benchmark is repeated local server-side execution only, not cold cache, full API, concurrent or distributed testing. Caches, fixed order and background activity can affect results; four cases and five trials per case do not characterize whole-catalogue latency distribution. Index build timing includes client overhead. Python syntax compilation and check-digit/fingerprint helper tests passed locally; live SQL/plan and benchmark execution unverified until user's output. Future I8.4 tests should exercise the actual joined application lookup and measure client/API latency separately.

### 7.19 I8.4a — Observed B-tree benchmark results (26 November 2026)

Both before and after phases completed on the user's Mac. Four verified fixtures: `00000105000011` (one row with NOVA), `00000105000059` (one row without NOVA), `00059527070430` (two matching rows), and `99999999990017` (zero local matches; generated structurally valid code, not a claimed assigned GTIN). Five measured queries per case per condition after warmup. All before plans were Seq Scan, all after plans Index Scan, and result SHA-256 fingerprints were unchanged. The index `idx_off_products_gtin14` is retained, nonunique, created in 4.835208 s client wall time; size 148,054,016 bytes (141.20 MiB).

| Case | Before median ms | After median ms | Ratio of medians | Before median shared hits / reads | After median shared hits / reads |
|---|---:|---:|---:|---:|---:|
| Known with NOVA | 1230.336 | 0.023 | 53,492.87 | 1920 / 718058 | 4 / 0 |
| Known without NOVA | 1263.176 | 0.023 | 54,920.70 | 1632 / 718346 | 4 / 0 |
| Ambiguous | 1268.360 | 0.026 | 48,783.08 | 1824 / 718154 | 5 / 0 |
| Locally absent | 1226.155 | 0.022 | 55,734.32 | 1728 / 718250 | 3 / 0 |

Raw trials (ms) in supplied order:

| Case | Before | After |
|---|---|---|
| Known with NOVA | 1223.147, 1215.394, 1237.154, 1278.316, 1230.336 | 0.024, 0.024, 0.023, 0.023, 0.023 |
| Known without NOVA | 1263.176, 1279.631, 1354.768, 1238.176, 1223.473 | 0.024, 0.023, 0.025, 0.023, 0.022 |
| Ambiguous | 1268.360, 1388.107, 1274.266, 1211.501, 1187.635 | 0.025, 0.026, 0.025, 0.031, 0.026 |
| Locally absent | 1226.155, 1249.013, 1281.875, 1220.305, 1192.289 | 0.025, 0.021, 0.022, 0.022, 0.021 |

Interpretation: the index avoids a full sequential scan for equality lookups. Shared buffer reads do not necessarily equal physical SSD reads; the OS cache can serve them. Warm indexed timings are at microsecond scale and rounding affects the large ratios. These are server execution times under EXPLAIN ANALYZE, not HTTP, Python client, model, mobile or whole-catalogue latency measurements. No cold-cache or concurrency claim. Do not rerun the before phase or drop the index. Evidence files `data/backend/i8_4a_index/{before.json,index_created.json,after.json}` are produced by the script; observed numeric results are from supplied terminal output. Actual table sizes were not included in that output.

### 7.20 I8.4b — Reusable application lookup and client timing (prepared)

Purpose: connect the future Python API directly to PostgreSQL rather than launching Spark for every scan. Install Psycopg 3 in `.venv-integration` using the official binary install command; capture the actual installed version, then pin it for reproduction. Do not change the Spark environment. Official installation reference: https://www.psycopg.org/install/. ARM64 binary wheels require macOS 14 or newer; actual installation is pending.

New modules `src/integration/catalogue_lookup.py` and `src/integration/test_i8_4_catalogue_lookup.py`. The first normalizes ASCII digit GTIN-8/12/13/14, checks the check digit, rejects all-zero/invalid values, and pads to 14 characters. This does not establish assigned GTIN identity and does not expand UPC-E. Use a parameter-bound SQL query selecting OFF rows and joining conflict and curated metadata by GTIN; preserve NULL recorded NOVA. Return `invalid_barcode`, `not_found` (local only), `ambiguous` (no chosen record), or `found`. Only singleton checked curated links (`consistent_checked_fields`, record_count=1) expose source comparison details; missing English comparison is visibly withheld, no substitution of USDA text or label. Even checked source linkage is not independent verification of formulation or NOVA. The existing generated hex password is read locally from ignored `infra/.env`; use a session with default transactions read-only and statement timeout. Existing admin account is a development shortcut, not a security boundary; create a least-privilege account before exposing an API. No data writes or model inference.

Test the four prior fixtures and compare projected fingerprints against the saved after.json, two leading-zero aliases, three invalid inputs and two dynamically selected curated link cases. Require identical alias results, correct recorded NOVA presence, and no arbitrary ambiguous record. Time six valid database cases with 30 calls each, shuffled order, one reused connection and prior warmup. Report median/min/max and raw trial times, connection setup separately, actual table sizes, and case fingerprints in `data/backend/i8_4b_lookup_report.json` (refuses overwrite). Python client timing includes normalization, local TCP, SQL, row decoding and decision logic, but not HTTP, model or mobile. Synthetic tests and syntax compilation passed in report-generation environment; actual Psycopg installation, connection, query and latency measurements not yet verified. The next step after successful I8.4b is I8.5 FastAPI.

```bash
.venv-integration/bin/python -m pip install 'psycopg[binary]'
.venv-integration/bin/python -m pip show psycopg psycopg-binary
.venv-integration/bin/python -m src.integration.catalogue_lookup
.venv-integration/bin/python -m src.integration.test_i8_4_catalogue_lookup
```

### 7.21 I8.4b — Observed application lookup results (26 November 2026)

Both the synthetic self-test and live application test passed on the user's Mac. Psycopg version 3.3.9; PostgreSQL server_version 170012 (17.12). Connection establishment measured once at 14.710750 ms. The original projected fixture hashes matched the I8.4a baseline, and the aliases returned identical full lookup results. No arbitrary record was returned for the ambiguous barcode. Missing NOVA was retained as NULL, and a missing-English source comparison was withheld as intended.

| Case | Input | Status | Rows |
|---|---|---|---:|
| Known with NOVA | `00000105000011` | found | 1 |
| Known without NOVA | `00000105000059` | found | 1 |
| Ambiguous | `00059527070430` | ambiguous | 2 |
| Locally absent | `99999999990017` | not_found | 0 |
| 12-digit alias | `000105000011` | found | 1 |
| 13-digit alias | `0000105000011` | found | 1 |
| Invalid check digit | `00000105000012` | invalid_barcode | 0 |
| Invalid format | `abc123` | invalid_barcode | 0 |
| All zero | `00000000` | invalid_barcode | 0 |
| Checked link | `00000000005487` | found | 1 |
| Missing English link | `00000000035071` | found | 1 |

All 11 cases passed. Six valid database cases were measured 30 times each (180 serial warmed calls); all repeated results matched their initial results according to the test. The supplied console output contains summary statistics, not the raw per-call timing arrays; these are retained in the local JSON report by the script.

| Case | Median ms | Minimum ms | Maximum ms |
|---|---:|---:|---:|
| Known with NOVA | 0.261792 | 0.222209 | 0.788708 |
| Known without NOVA | 0.247229 | 0.188541 | 0.837000 |
| Ambiguous | 0.3716665 | 0.298125 | 0.689167 |
| Locally absent | 0.229562 | 0.186333 | 0.345333 |
| Checked link | 0.2701045 | 0.223959 | 0.634709 |
| Missing English link | 0.2777295 | 0.215459 | 0.478792 |

| Table | Total bytes (pg_total_relation_size) |
|---|---:|
| off_products | 7,207,714,816 |
| curated_links | 297,189,376 |
| barcode_conflicts | 32,768 |
| Combined | 7,504,936,960 |

These sizes include table storage, indexes and auxiliary storage, not the entire Docker volume, WAL or demonstration schemas. The latency measurements include barcode validation, SQL, local TCP, result conversion and classification; exclude connection creation, HTTP, inference and Android. They are not a concurrency or cold-cache benchmark and do not characterize the complete catalogue. The existing admin credential with a read-only session was a local prototype shortcut; a restricted account is still required before API deployment. Connection pooling is proposed for the API, not yet implemented. Curated-link consistency is not independent formulation or NOVA validation. Raw JSON evidence: `data/backend/i8_4b_lookup_report.json`.

### 7.22 Git checkpoint — recommended, not yet verified

The user asked whether it is time to push to GitHub. Yes: I8.4b is a coherent verified checkpoint. Commit the integration Python scripts, current documentation, safe Compose configuration, ignore rules and dependency pins (observed Psycopg 3.3.9). Preserve both failed and corrected loading scripts for reproducibility and use the revised loader for this snapshot. Keep credentials, local virtual environments, generated datasets, model weights, database files and raw experiment outputs outside Git. Reports record observed results; retain local raw evidence separately. Inspect branch and remote, check ignores and tracked secret paths, stage selected files rather than `git add .`, inspect the staged diff, then commit and push `feature/bda-dl-integration` to `origin` without force. Do not merge into main or the mobile branch. The current local Git state and remote divergence are not visible here; no commit or push is claimed. Next engineering step I8.5: restricted PostgreSQL account and local FastAPI product lookup, then DL integration.

### 7.23 Git checkpoint outcome and I8.5a/b — restricted account and HTTP API (prepared)

The user confirmed a successful push after reviewing 15 staged files. An unrelated untracked MAD lab project was identified and advised to be moved outside the repository. The exact remote commit and final status are not supplied. Keep the integration branch separate from main and mobile. Commit after each verified milestone.

I8.5a: `api_db_access.py` provisions a new `food_api` login with SELECT on exactly three catalogue tables, USAGE on catalogue schema, CONNECT on foodfacts, no administrator attributes, default read-only transactions and ten-second statement timeout. A generated 48-hex password is kept in ignored `infra/.env.api` (0600); do not upload it. The admin credentials are used only for provisioning. A libpq SCRAM verifier is used for CREATE ROLE, not a plaintext password SQL literal. Verification checks SELECT and absence of named table write privileges, no schema CREATE, role attributes and two actual lookup cases. It refuses to overwrite an existing secret or role. This is a local restricted catalogue login, not comprehensive database hardening; public privileges on unrelated objects may still exist. No catalogue data mutation is intended.

I8.5b: `food_api.py` reuses `catalogue_lookup.lookup()` and its established statuses. Uvicorn serves on 127.0.0.1:8000, one process. Lifespan opens/closes a Psycopg pool (1–4 connections, 5-second acquisition timeout); synchronous routes borrow connections. `/health` checks a nonempty catalogue table. `/products/{barcode}` returns 200 found, 400 invalid, 404 locally absent, 409 ambiguous; database failures 503 and consistency error 500. Status bodies retain the existing lookup contract, including nullable recorded NOVA and conservative curated-link display. No automatic DL inference, Kafka events, auth or Android networking yet. The API returns the current lookup projection; field minimization and stable mobile DTOs will be reviewed before LAN exposure. Bind loopback only.

`test_i8_5_http.py` makes actual HTTP calls to the server, checks health and the same 11 lookup fixtures, confirms alias equivalence, NOVA NULL preservation and source comparison withholding. Saves a fresh JSON report to `data/backend/i8_5b_http_report.json` with observed package versions; refuses overwrite. No controlled HTTP timing yet. Install fastapi, uvicorn and psycopg-pool in `.venv-integration` without upgrading existing Psycopg; pin versions after verified installation. Python syntax compilation passed; live provisioning, pool, HTTP and role tests remain pending. Official references: https://www.postgresql.org/docs/17/sql-grant.html ; https://www.psycopg.org/psycopg3/docs/api/pq.html ; https://www.psycopg.org/psycopg3/docs/advanced/pool.html ; https://fastapi.tiangolo.com/advanced/events/ .

### 7.24 I8.5b — live API server observations (partial verification)

Supplied Uvicorn server output shows successful startup on loopback `127.0.0.1:8000`, successful application lifespan startup and shutdown, repeated HTTP 200 health checks, and HTTP 200 responses for `/docs` and `/openapi.json`. The browser's optional `/favicon.ico` returned 404; this is unrelated to product lookup functionality. The server was stopped by Ctrl+C and completed graceful shutdown.

| Product barcode | Case | Observed HTTP status |
|---|---|---:|
| 00000105000011 | Known with NOVA | 200 |
| 00000105000059 | Known without NOVA | 200 |
| 00059527070430 | Ambiguous | 409 |
| 99999999990017 | Locally absent | 404 |
| 000105000011 | 12-digit alias | 200 |
| 0000105000011 | 13-digit alias | 200 |
| 00000105000012 | Invalid checksum | 400 |
| abc123 | Invalid format | 400 |
| 00000000 | All zero | 400 |
| 00000000005487 | Checked link | 200 |
| 00000000035071 | Missing English comparison | 200 |

All 11 expected status codes match. Access logs do not prove response-body assertions, role privileges or successful creation of the client report. The existing server code uses the `food_api` connection configuration and pool startup succeeded; separate least-privilege checks still need their output. Use `.venv-integration/bin/python -m src.integration.api_db_access --verify-only` rather than recreate the role. Inspect `data/backend/i8_5b_http_report.json` if already generated. If absent, restart Uvicorn on loopback and run `test_i8_5_http` from a separate terminal, which validates the JSON response contract and writes the report without overwriting an existing one. The server log alone does not establish whether the client assertions after the last request passed. Do not claim the automated suite fully passed until its report or success output is supplied. Package versions other than previously observed Psycopg 3.3.9 are not yet confirmed. No HTTP latency, concurrency, database outage, DL inference or Android networking test claimed. Next: finish these checks, record dependency pins, commit the verified API milestone, then I8.6 model integration.

### 7.25 I8.5a/b — final permission and HTTP client verification (completed)

User-supplied terminal output from `.venv-integration/bin/python -m src.integration.api_db_access --verify-only` confirms SELECT permitted and named write/other privileges denied on `catalogue.off_products`, `catalogue.curated_links`, and `catalogue.barcode_conflicts`; the script's role attribute, read-only transaction, schema CREATE denial, known and ambiguous lookup assertions completed successfully. This is restricted catalogue access, not a comprehensive production security audit.

The user supplied `data/backend/i8_5b_http_report.json`. Health returned `{"status":"ok","database":"ready"}`. All 11 cases passed, with the case/status/row counts below. Inspection of the test script confirms that report generation follows assertions on the product selection contract, recorded NOVA presence and NULL preservation, equality of the 12-/13-digit alias responses, checked source comparison visibility and missing English comparison withholding.

| Case | HTTP | Lookup status | Matching rows |
|---|---:|---|---:|
| known | 200 | found | 1 |
| missing_nova | 200 | found | 1 |
| ambiguous | 409 | ambiguous | 2 |
| absent | 404 | not_found | 0 |
| alias_12 | 200 | found | 1 |
| alias_13 | 200 | found | 1 |
| bad_checksum | 400 | invalid_barcode | 0 |
| bad_format | 400 | invalid_barcode | 0 |
| zero | 400 | invalid_barcode | 0 |
| checked_link | 200 | found | 1 |
| missing_english | 200 | found | 1 |

Observed dependencies for the direct backend requirements file: `fastapi==0.165.0`, `uvicorn==0.54.0`, `psycopg[binary]==3.3.9`, `psycopg-pool==3.3.3`. These are direct dependency pins, not a complete transitive lockfile. Apply to `requirements-backend.txt`, check `pip check`, verify both credential paths are ignored and untracked, stage `.gitignore`, the requirements file, `api_db_access.py`, `food_api.py`, `test_i8_5_http.py`, and this report. Review staged diff, commit, push only `feature/bda-dl-integration`. Previous batch checkpoint was user-confirmed; new API checkpoint is pending. No model weights, passwords or generated database files should be committed.

Limitations: local serial HTTP smoke test, no measured HTTP latency or concurrency, no outage/recovery test, no authentication or production deployment, no DL or Android integration. I8.5 is closed at the specified local smoke-test scope.

### 7.26 I8.6a — existing model preflight (completed)

Before modifying the verified API, check the four checkpoint file SHA-256 hashes against `docs/NOVA_INFERENCE_POLICY.json`, load the existing model using `src.deep_learning.inference.predict.load_model()` in `.venv-integration`, and call `predict_nova()` on simple example ingredient lists. Verify output class range and print device, load time and model output. This is an environment/checkpoint compatibility smoke test, not a new model accuracy experiment. Reuse the original inference mode (alpha 0, temperature 1); research-only corrections are not exposed to general requests. No retraining. Keep recorded NOVA separate from predicted NOVA, load model once in server lifespan and serialize initial inference for controlled local testing. Planned POST /predict accepts ingredient text; OCR review and additional input safeguards are required before mobile exposure. Do not modify the existing HTTP lookup contract until regression tests are prepared. Actual preflight supplied by the user: config.json, model.safetensors, tokenizer.json and tokenizer_config.json all MATCH against policy SHA-256 hashes. Original load_model/predict_nova executed on mps:0; model load 1.477973375 s. Four smoke inputs:

| Ingredient text | Predicted NOVA | Scores NOVA 1/2/3/4 (%) | Inference ms |
|---|---:|---|---:|
| Milk | 1 | 96.84 / 1.71 / 1.22 / 0.23 | 217.826583 |
| Sunflower oil | 2 | 0.17 / 98.95 / 0.81 / 0.07 | 15.935833 |
| Milk, salt, bacterial culture, rennet | 3 | 0.24 / 1.09 / 98.37 / 0.30 | 14.920458 |
| Wheat flour, sugar, palm oil, emulsifier, artificial flavouring | 4 | 0.33 / 0.41 / 2.06 / 97.21 | 13.900417 |

All four outputs in range. The first inference was slower than subsequent calls; cause not isolated. Rounded scores can sum to 100.01. These are illustrative functionality tests, not independent accuracy or a latency benchmark. No retraining. User provided `nothing to commit, working tree clean`; verify upstream and dependency pins separately rather than infer successful remote push from clean status.

### 7.27 I8.6b — original-mode prediction HTTP endpoint (prepared, live execution pending)

Files: `src/integration/nova_service.py`, revised `src/integration/food_api.py` (distributed as `food_api_v2.py`), and `src/integration/test_i8_6_prediction_http.py`. Preserve the existing FastAPI GET handlers and connection pool. Startup verifies policy labels/default mode and the four checkpoint hashes, then calls original load_model once. Model failure leaves catalogue API available and prediction returns 503. Single Uvicorn worker, local loopback, lock serializes tokenizer/model access. No new model libraries needed in the existing integration environment; use installed Pydantic v2 and record version. Do not expose research adjustments or alter model or database. POST /predict body: ingredient_text string, source manual (default) or ocr, reviewed boolean (default false). Unknown fields rejected. Basic guards reject blank/symbol-only/control text, unreviewed OCR, >4000 characters, >256 tokenizer tokens including special tokens (no silent truncation). Legitimate one-ingredient examples allowed; arbitrary alphabetic OCR fragments still possible. User review is a client assertion, not independently verified. 400 domain validation, 422 malformed/invalid body, 503 model unavailable; 200 returns original scores, predicted_nova, label, confidence, model/policy/source/token metadata and caution. Recorded NOVA not returned or overwritten by prediction. No confidence threshold, no OCR image processing, no language detector or robust fragment guard yet; I8.7 remains necessary. No full request-size limit/rate limiter/authentication; local testing only.

Test procedure: run `python -m src.integration.nova_service` in integration environment, start Uvicorn with `src.integration.food_api:app --host 127.0.0.1 --port 8000`, then `python -m src.integration.test_i8_6_prediction_http` in separate terminal. It checks health and 11 original lookup cases including recorded-label and curated-link semantics; compares all four manual predictions with observed I8.6a scores within 0.1 percentage point and exact class; tests a reviewed synthetic OCR text and nine rejected request cases; checks lookup unchanged after prediction. Exclusive output `data/backend/i8_6b_prediction_http_report.json`. The tolerance and caps are engineering choices, not validated statistical thresholds. Python compilation and pure ingredient validation self-tests passed in assistant environment; no local checkpoint, PostgreSQL or full HTTP test available here. Live Mac tests pending. Reference: https://fastapi.tiangolo.com/advanced/events/ and https://fastapi.tiangolo.com/tutorial/body/.

### 7.28 I8.6b — completed HTTP integration results

User-supplied `test_i8_6_prediction_http` output confirms 11 product cases, four manual predictions, one reviewed synthetic OCR example and nine invalid cases (25 total), health and final lookup unchanged. Observed FastAPI 0.165.0, Pydantic 2.14.4, torch 2.15.0, transformers 5.24.0 and Uvicorn 0.54.0. The four returned vectors exactly equal the earlier rounded scores in 7.26. Model mode original, policy 1.0. Token counts include special tokens.

| Ingredient | NOVA | Confidence (%) | Tokens | HTTP ms |
|---|---:|---:|---:|---:|
| Milk | 1 | 96.84 | 3 | 105.113125 |
| Sunflower oil | 2 | 98.95 | 5 | 16.908791 |
| Milk, salt, bacterial culture, rennet | 3 | 98.37 | 11 | 17.118375 |
| Wheat flour, sugar, palm oil, emulsifier, artificial flavouring | 4 | 97.21 | 17 | 16.643375 |

Last-three HTTP mean 16.890180 ms; single serial call per example, not a controlled latency benchmark. OCR fixture Milk returned the same vector with source ocr. Blank, symbols, unreviewed OCR, control character, excessive tokens returned 400 invalid_ingredients. Excessive characters, missing text, numeric text and research_mode returned 422 invalid_request. These are functionality tests, not independent accuracy or real-image OCR evaluation. Existing model unchanged. Git checkpoint proposed: stage `src/integration/nova_service.py`, `src/integration/food_api.py`, `src/integration/test_i8_6_prediction_http.py`, canonical report and any deliberately updated requirements; check staged contents and credentials, commit and push integration branch. No claim of completed push yet.

### 7.29 I8.7a — offline ingredient-quality triage (prepared)

Previous I6.6 research documented high scores for `aoa` and `=`; headings `Ingredients` and `ingredients\n\nAM` were also unusable. Reuse the actual previously saved `data/integration/evaluation_i6_6/i6_6_ocr_metrics.json` (12 attempts, six images, two PSM values) and six unique corresponding reference texts. Do not rerun OCR or the model. Existing API stays unchanged during pilot. Add `src/integration/ingredient_quality.py` and `src/integration/test_i8_7_ingredient_quality.py`. Existing basic validator runs first. Heading without alphabetic content is rejected. Initial heading stripped only for inspection, never from model input. Possible packaging metadata headings and unknown fragments with at most three alphabetic characters are flagged for review; small common ingredient allowlist permits e.g. tea, oil, dal, rai. Remaining text accepted provisionally; this is not a food dictionary or completeness guarantee. Do not use confidence as a quality gate. No automatic ingredient rewrite, label change or model retraining.

Run `.venv-integration/bin/python -m src.integration.test_i8_7_ingredient_quality` from repository root. 24 designed fixtures passed in assistant environment using existing basic validator. Mac run reads previous OCR JSON, verifies all 12 barcode/PSM combinations and checks documented failure cases and six reference texts. It compares baseline vs candidate decisions with reviewed=True simulated for both; unreviewed OCR remains rejected in both. Output exclusive `data/backend/i8_7a_ingredient_quality_report.json`, records prior report SHA-256, per-case decisions, prior OCR WER/prediction/score and counts without raw ingredient text. Known bad examples are development regression tests, not independent validation. If old OCR report missing, recover existing file or report missing; do not fabricate it. No new OCR, model inference or DB writes. Future I8.7b: use observed results to decide API enforcement and retake/edit UX; preserve existing contract and token limit and regression tests. Unknown short legitimate regional words may be flagged; longer gibberish and wrong-language text can pass. Small allowlist, metadata patterns and three-letter heuristic are engineering decisions, not validated universal thresholds. Real OCR replay remains pending.

## 8. Planned API and event safety

Proposed endpoints: `GET /health`, `GET /products/{barcode}`, `POST /predict`, later `POST /scan-events`. The API contract must distinguish invalid, absent, ambiguous and found barcodes. Missing local data does not establish that the product does not exist elsewhere. OCR/manual input must be reviewed, reject unusable symbols/fragments without excluding legitimate single ingredients, and retain separate recorded and AI classifications. Scores are not guarantees. The current model is not newly validated on India/OCR inputs by the holdout improvement.

Use a minimal consent-aware event schema with event ID, timestamp, barcode where permitted, lookup/prediction outcome and provenance; do not collect personal identifiers, precise location or raw label images by default. Keep event schema version, timestamps and event IDs explicit; plan idempotent writes, checkpoint recovery, duplicate handling, late-event behavior, retention and access controls. Kafka/streaming recovery semantics must be tested; no end-to-end exactly-once or production scalability claim is made. Store credentials in ignored local environment configuration, not source control or ChatGPT project files. Bind development services locally unless a controlled test requires LAN access. Distinguish synthetic load tests from real mobile scan events and measure actual dashboard data.

## 9. Report maintenance and revision history

Use `docs/BACKEND_DEVELOPMENT_REPORT.md` as the canonical current copy. For each completed step, record a plain-English purpose, exact commands and versions, source/derived artifact paths, observed outputs and metrics, failures, decisions and limitations. Keep generated datasets, database volumes, logs containing sensitive information, credentials and model weights out of Git. Commit source scripts, safe configuration examples and the reports. Preserve older revisions in Git history. The research report is closed through I7 unless an explicit correction or future DL study is needed.

| Revision | Date | Change |
|---|---|---|
| 0.1 | 2026-11-26 | Carried forward I8.1–I8.2 observations from former combined revision 0.14; retired unexecuted SQLite builder; recorded revised Big Data architecture and first preflight. |
| 0.2 | 2026-11-26 | Recorded environment results, Java 17 and separate environment decision; Docker setup and Spark smoke test planned. |
| 0.3 | 2026-11-26 | Docker and Spark smoke tests passed; local PostgreSQL service and isolated SQL/persistence exercise planned. |
| 0.4 | 2026-11-26 | PostgreSQL demo user-confirmed without detailed logs; pinned JDBC dependency and round-trip test prepared. |
| 0.5 | 2026-11-26 | JDBC round trip verified; bounded Spark real-data catalogue pilot and staging design prepared. |
| 0.6 | 2026-11-26 | Real-data pilot completed and reconciled; full-snapshot ETL and global quality audit prepared. |
| 0.7 | 2026-11-26 | Full snapshot ETL verified; real-data JDBC pilot prepared. |
| 0.8 | 2026-11-26 | Real-data JDBC pilot verified; full catalogue loading protocol and script prepared. |
| 0.9 | 2026-11-26 | Recorded safe full-load preflight stop on literal NUL; prepared explicit audited compatibility correction; full database load pending. |
| 0.10 | 2026-11-26 | Audited NUL correction and full JDBC load verified; prepared I8.4a controlled lookup comparison. |
| 0.11 | 2026-11-26 | Recorded actual before/after indexing results; prepared read-only joined application lookup and client benchmark. |
| 0.12 | 2026-11-26 | Recorded successful 11-case application lookup, 180 measured calls, table sizes and recommended Git checkpoint before I8.5. |
| 0.13 | 2026-11-26 | User-confirmed Git push; prepared restricted database role, pooled FastAPI and live HTTP smoke test. |

| 0.14 | 2026-11-26 | Recorded live API startup, expected HTTP status codes for 11 product fixtures, documentation and shutdown; role and client-body verification pending. |

| 0.15 | 2026-11-26 | Completed restricted-role and HTTP client verification; observed dependency pins and planned API Git checkpoint and model preflight. |

| 0.16 | 2026-11-26 | Recorded verified checkpoint hashes and MPS smoke outputs; prepared original-mode prediction API, basic input checks and HTTP regression tests. |

| 0.17 | 2026-11-26 | Verified I8.6b HTTP test results; prepared offline I8.7a quality triage and Git checkpoint. |

**Evidence boundary:** Full catalogue loading, correction and reconciliation are supported by supplied terminal output, not independently rerun here. Persistence exercise is user-confirmed without separate logs. Index benchmark and direct Python lookup are supported by supplied terminal output; Git push is user-confirmed (no remote commit hash supplied); live API server checks, restricted-role checks and HTTP client assertions are supported by the supplied logs and report. API checkpoint clean working tree is observed, but remote push and dependency pins are not independently confirmed. I8.6a preflight is supported by supplied output; I8.6b live HTTP verification is supported by supplied output. I8.7a actual OCR replay, API enforcement, Android and Kafka remain pending.
