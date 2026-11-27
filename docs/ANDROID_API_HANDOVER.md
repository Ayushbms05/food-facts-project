# Food Facts — Android API handover

**I9.1 | 27 November 2026 | Backend API 0.3.0 | Integration contract, not an Android implementation**

Audience: Member 3 (Kotlin/XML). Backend: `feature/bda-dl-integration`. Mobile: `member3-kotlin-xml` (latest remote contents still to inspect). Do not merge or overwrite mobile code until agreed. Backend host runs Docker PostgreSQL, native FastAPI and the original DistilBERT on MPS. No database credentials, model files or raw datasets belong in the Android app.

## 1. Local connectivity

On the backend Mac, from repository root:

```sh
docker compose -f infra/compose.yaml ps
.venv-integration/bin/python -m uvicorn src.integration.food_api:app --host 127.0.0.1 --port 8000
```

If the API is already running, reuse it. From a second terminal:

```sh
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/products/00000105000011
```

Android Emulator on the **same Mac**: `http://10.0.2.2:8000`. Open `/health` and `/products/00000105000011` in the emulator browser first. On Android, 127.0.0.1 refers to Android itself. The 10.0.2.2 alias is specific to the emulator's host, not another team member's laptop. Physical device/remote teammate connection is a separate step; do not open PostgreSQL or bind an unauthenticated API to the college network. A debug-only HTTP exception may be needed for Android app requests; keep production HTTPS requirements intact. Inspect existing manifest and security configuration first. App needs INTERNET permission. Avoid blocking UI thread; use the app's existing coroutine/network architecture if present. Make base URL configurable for debug builds.

Official references: https://developer.android.com/studio/run/emulator-networking-address ; https://developer.android.com/privacy-and-security/security-config .

## 2. Endpoints and handling

| Request | HTTP | JSON status | Client action |
|---|---:|---|---|
| GET /health | 200 | ok | Database ready (not model readiness) |
| GET /products/{barcode} | 200 | found | Display available product data |
| GET /products/{barcode} | 400 | invalid_barcode | Ask for correct barcode |
| GET /products/{barcode} | 404 | not_found | Local absence only; offer OCR/manual entry |
| GET /products/{barcode} | 409 | ambiguous | Do not choose automatically; offer manual/OCR path |
| GET /products/{barcode} | 503 | database_unavailable | Retry later; manual prediction may still work |
| GET /products/{barcode} | 500 | catalogue_inconsistent | Report service issue; do not treat as not found |
| POST /predict | 200 | predicted | Display predicted NOVA and model scores separately |
| POST /predict | 400 | invalid_ingredients | Correct/re-enter ingredients |
| POST /predict | 400 | needs_review | Inspect and correct suspicious text; no prediction yet |
| POST /predict | 422 | invalid_request | Fix request format |
| POST /predict | 503 | model_unavailable | Prediction unavailable; lookup may still work |

Transport timeout or connection failure is not a barcode miss. Parse HTTP status and JSON `status` (including non-2xx response bodies). Do not automatically retry permanently invalid input. Distinguish loading, success, missing data and error states. No authentication, production deployment or analytics endpoint is implemented in 0.3.0.

### Barcode lookup

Barcodes are **strings**, preserving leading zeros. Backend checks ASCII digits of length 8/12/13/14, nonzero, GS1 check digit and normalizes to GTIN-14. No UPC-E expansion. Structural validity does not prove product identity. `found` means one matching local record. Abridged example (other fields omitted):

```json
{"status":"found","gtin14":"00000105000011","matching_rows":1,"product":{"product_name":"Chamomile Herbal Tea","recorded_nova":1},"curated_link_status":"...","source_comparison":null}
```

The ellipsis and null comparison here are placeholders, not the actual observed comparison for this fixture. Inspect live JSON before creating DTOs. Source code returns all OFF product columns, excluding internal join fields, including `record_id`, `gtin14`, `off_code`, `product_name`, `brands`, `ingredients_raw`, `ingredients_en`, `ingredients_structured_json`, `recorded_nova`, `country_tags_json`, `allergen_tags_json`, `trace_tags_json`, `additive_tags_json`, `nutrients_json`, and others. Fields may be null. Fields ending in `_json` and `ingredients_structured_json` are serialized JSON **strings** (or null), not native nested arrays; parse carefully. `source_comparison` is a nested object or null, exposed only for checked singleton links; it is not proof of identical formula. Missing allergen metadata does not prove absence. Recorded NOVA is nullable, not zero. Do not equate NOVA with a health score.

For ambiguous/not_found/invalid responses there is no `product` key. Normalized `gtin14` is null for invalid barcodes. No automatic inference on lookup; do not silently use USDA comparison-cleaned text or overwrite recorded NOVA. For prediction, explicitly review the original ingredient list (English preferred; model not validated for arbitrary languages). If only partial/uncertain text exists ask user to correct it. Exact structured ingredient extraction parity is deferred; do not feed raw serialized JSON to the model.

### NOVA prediction

POST JSON:

```json
{"ingredient_text":"Milk","source":"manual","reviewed":false}
```

`source` is `manual` or `ocr`. For OCR require actual user review/edit, then send `reviewed:true`. Max 4000 characters and 256 tokenizer tokens including special tokens. Extra fields rejected; no research adjustment mode. Do not silently truncate. Successful observed response:

```json
{"status":"predicted","predicted_nova":1,"label":"NOVA 1","confidence":96.84,"class_probabilities":{"NOVA 1":96.84,"NOVA 2":1.71,"NOVA 3":1.22,"NOVA 4":0.23},"prediction_mode":"original","model":"distilbert_food_nova","policy_version":"1.0","input_source":"manual","token_count":3,"confidence_note":"Model score, not a guarantee of correctness; not a health rating."}
```

Scores are percentages, not calibrated guarantees. Ingredient-quality review response has reason, message and rule version when applicable. Other invalid responses can omit these fields. Do not render a prediction if no `predicted_nova` exists. Rule version 0.1 is heuristic and incomplete; reviewed flag alone is not evidence of real review. Avoid logging raw ingredient text or pictures; no event submission until consent-aware event contract is designed.

## 3. Acceptance fixtures

| Input | Expected |
|---|---|
| 00000105000011 | Found; NOVA 1 |
| 00000105000059 | Found; recorded NOVA null |
| 00059527070430 | Ambiguous, two rows |
| 99999999990017 | Not found in this snapshot |
| abc123 | Invalid barcode |
| 000105000011 | Same product as first fixture |
| 00000000005487 | Found; source comparison available |
| 00000000035071 | Found; source comparison withheld |
| Manual Milk | NOVA 1, observed 96.84% |
| OCR aoa, reviewed true | Needs review, no prediction |
| OCR Milk, reviewed false | Invalid ingredients |
| OCR Milk, reviewed true | Predicted NOVA 1 |

First prove health/product from emulator browser, then implement network adapter and screens on a separate agreed mobile integration branch. Test UI paths, no-network behavior, reviewed OCR, and record results. No live Android testing has occurred at this point.
