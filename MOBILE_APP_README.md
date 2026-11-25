# Food Facts — Mobile Application Documentation (Member 3)

## Project Overview
The **Food Facts** mobile application is a native Android application built using **Kotlin** and **Android XML layouts** (Traditional Views). It serves as the mobile user interface for an integrated college project combining:
1. **Big Data Analytics (Member 1)**: Curated database integration (Open Food Facts + USDA FoodData Central) & GTIN-14 scan event streaming pipeline.
2. **Deep Learning (Member 2)**: Ingredient-text NOVA food processing group (1 to 4) prediction service.
3. **Mobile Application Development (Member 3)**: Kotlin + XML Views application providing barcode scanning, product search, ingredient OCR, product comparison, Room scan history, and integration layers for team components.

---

## Android Technology Stack
- **Language**: Kotlin 2.0
- **UI Toolkit**: Android XML Layouts, Material Design 3 Components, Traditional Views, Activities, Fragments
- **Architecture**: Lightweight Model-View-ViewModel (MVVM)
- **View Binding**: Android ViewBinding (No Jetpack Compose)
- **Navigation**: Android Jetpack Navigation Component (`NavHostFragment`, `nav_graph.xml`)
- **Database & Local Persistence**: Room Database (`AppDatabase`, `ProductDao`, `ScanEventDao`)
- **Networking & Serialization**: Retrofit 2, OkHttp 3 with Logging Interceptor, Gson
- **Camera & Scanning**: CameraX (`camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view`), Google ML Kit Barcode Scanning
- **OCR Text Recognition**: Google ML Kit Text Recognition
- **Image Loading**: Coil (`io.coil-kt:coil`)
- **Asynchronous & State**: Kotlin Coroutines, StateFlow, LiveData, Lifecycle-Ktx

---

## Project Structure & Architecture
```
app/src/main/java/com/example/foodfacts/
├── MainActivity.kt
├── data/
│   ├── api/
│   │   ├── model/ (ApiProduct, ApiNutriments, ProductResponse, SearchResponse)
│   │   ├── NetworkClient.kt
│   │   └── OpenFoodFactsApi.kt
│   ├── local/
│   │   ├── AppDatabase.kt
│   │   ├── ProductDao.kt
│   │   ├── ProductEntity.kt
│   │   ├── ScanEventDao.kt
│   │   └── ScanEventEntity.kt
│   ├── model/
│   │   ├── ComparisonResult.kt
│   │   ├── MetricComparison.kt
│   │   ├── ProductDomainModel.kt
│   │   ├── Resource.kt
│   │   └── ScanEvent.kt
│   └── repository/
│       └── ProductRepository.kt
├── integration/
│   ├── events/
│   │   └── ScanEventLogger.kt (Member 1 Big Data Event Logger)
│   └── prediction/
│       ├── DeepLearningApi.kt (Member 2 DL Model REST API)
│       ├── DeepLearningRepository.kt
│       └── PredictionModels.kt
├── ui/
│   ├── adapters/ (MetricAdapter, ProductAdapter, ScanEventAdapter)
│   ├── fragments/
│   │   ├── CompareFragment.kt
│   │   ├── HistoryFragment.kt
│   │   ├── HomeFragment.kt
│   │   ├── OcrFragment.kt
│   │   ├── ProductDetailFragment.kt
│   │   ├── ScanFragment.kt
│   │   └── SearchFragment.kt
│   └── viewmodels/
│       ├── CompareViewModel.kt
│       ├── HistoryViewModel.kt
│       ├── OcrViewModel.kt
│       ├── ProductDetailViewModel.kt
│       ├── ScanViewModel.kt
│       └── SearchViewModel.kt
└── util/
    ├── BarcodeAnalyzer.kt
    └── OcrHelper.kt
```

---

## Core Features & Functionality

### 1. Home Dashboard (`HomeFragment`)
- Direct navigation buttons for all 5 core workflows:
  - 📷 **Scan Product Barcode**
  - 🔍 **Search Products by Name**
  - 📝 **Scan & Extract Ingredients (OCR)**
  - ⚖️ **Compare Two Products**
  - 📜 **Scan History & Event Logs**
- Overview of team integration status (Big Data + Deep Learning).

### 2. Barcode Scanning (`ScanFragment`)
- Real-time camera barcode scanner powered by CameraX and ML Kit.
- Manual barcode entry option.
- GTIN-14 normalization preserving leading zeros (e.g. `003017620422003`).
- Rapid scan debouncing (3-second window) to prevent duplicate triggers.
- Graceful error handling for missing/unknown products or network dropouts.

### 3. Product Search (`SearchFragment`)
- Real-time product search querying Open Food Facts API.
- Debounced user typing (500ms) with search action IME handling.
- RecyclerView results displaying product name, brand, barcode, Nutri-Score, and image.
- Empty result, loading, and connection error handling.

### 4. Product Details (`ProductDetailFragment`)
- Displays product name, brand, barcode, high-res image, and data source.
- Displays badges for Nutri-Score (A-E), NOVA Processing Group (1-4), and Eco-Score.
- Explains the distinction between **NOVA Industrial Processing (1-4)** and **Nutri-Score Nutritional Quality (A-E)**.
- Complete nutritional facts per 100g: Calories, Fat, Saturated Fat, Carbohydrates, Sugars, Dietary Fiber, Proteins, Salt.
- Shows `"N/A"` for missing values without fabricating zeros or artificial numbers.
- List of ingredients, declared allergens, and food additives.
- Integrated **Member 2 AI Model NOVA Predictor button**.

### 5. Ingredient OCR (`OcrFragment`)
- Capture product label photograph using camera or select image from gallery.
- Image preview and text extraction using ML Kit Text Recognition.
- **Editable ingredients input field** allowing users to refine OCR output or manually type ingredient text.
- Automatic allergen detection highlighting milk, wheat, soy, peanuts, tree nuts, etc.
- **Predict NOVA with AI Model button** for Member 2 integration.
- **Search Open Food Facts Database** using extracted ingredient terms.
- Reset button to clear preview and start a new scan.

### 6. Product Comparison (`CompareFragment`)
- Side-by-side comparison of two products selected from scan history.
- Metric-by-metric analysis of Nutri-Score, NOVA Group, Energy, Sugars, Saturated Fat, Salt, Fiber, and Proteins.
- Computes metric winner and summary without fabricating unsupported single overall health scores.
- Allows changing selections or clearing comparison.

### 7. Scan History & Event Logging (`HistoryFragment`)
- Persistent local scan history powered by Room Database.
- **Cached Products Tab**: Reopen previously scanned products offline. Delete individual items or clear history with confirmation dialog.
- **Scan Events Tab**: Log of all scan events containing GTIN-14 normalized barcodes, scan type, timestamps, and lookup status.

---

## Team Integration Layer Details

### Big Data Integration (Member 1)
- `ScanEventLogger` records every product lookup with fields matching the Spark Streaming pipeline requirements:
  - `event_id`: Auto-generated unique ID
  - `barcode` & `normalized_barcode`: Raw code and GTIN-14 (14-digit padded string)
  - `product_found`: Boolean flag
  - `nova_group` & `nova_source`: Open Food Facts / DL Model / None
  - `scan_type`: `BARCODE`, `OCR`, or `MANUAL_SEARCH`
  - `lookup_status`: `KNOWN_WITH_NOVA`, `KNOWN_WITHOUT_NOVA`, `UNKNOWN_PRODUCT`, or `NETWORK_FAILURE`
- Events are stored locally in Room DB (`scan_events`) until Member 1's ingestion endpoint is active.

### Deep Learning Integration (Member 2)
- `DeepLearningRepository` provides a REST API client (`POST /api/v1/predict-nova`) with configurable base URL (default: `http://10.0.2.2:8000/`).
- Sends `ingredients_text` to the model.
- Parses response: `predicted_nova_group`, `confidence_score`, `ingredient_evidence`, `explanation`, and `model_version`.
- When backend service is offline, it displays a truthful message: *"AI prediction service is not connected (http://10.0.2.2:8000). Member 2's Deep Learning service is offline."*

---

## Build & Test Instructions

### Build Project
```bash
./gradlew assembleDebug
```

### Run Unit Tests
```bash
./gradlew testDebugUnitTest
```

### Run Android Lint
```bash
./gradlew lintDebug
```

---

## Testing & Verification Summary
- **Compilation**: Clean build using `./gradlew assembleDebug`.
- **Unit Testing**: All 11 unit tests passed (`:app:testDebugUnitTest`).
- **Feature Verification**:
  - `HomeFragment` navigation buttons wired.
  - `ScanFragment` CameraX preview & barcode detection active.
  - `SearchFragment` Open Food Facts network search verified.
  - `ProductDetailFragment` displays complete nutrition, NOVA/Nutri-Score, additives & AI prediction card.
  - `OcrFragment` ML Kit text extraction & editable input verified.
  - `CompareFragment` metric calculation & history selection verified.
  - `HistoryFragment` Room database persistence verified.
