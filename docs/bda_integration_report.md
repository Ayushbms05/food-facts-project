
# Big Data Integration Report

## Data Sources

1. Open Food Facts
   - Format: Parquet
   - Total records: 4,893,632

2. USDA FoodData Central
   - Format: CSV
   - Branded food records: 2,010,212

## Integration Method

- Used Apache Spark.
- Standardized barcode representations.
- Validated barcode check digits.
- Selected the latest published USDA record per barcode.
- Joined datasets using normalized GTIN identifiers.

## Results

- Matched products: 368,055
- Products with OFF English ingredients: 314,306
- Products with USDA ingredients: 365,741
- Products with ingredients from both: 313,501

## Important Findings

USDA contains multiple records for many barcodes.

Matching barcodes do not guarantee identical ingredient
lists or product formulations.

Further ingredient comparison and validation are needed.

## Next Work

- Compare ingredient information across sources.
- Measure missing and conflicting values.
- Prepare source-aware integrated datasets.
