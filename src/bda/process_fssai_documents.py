
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SOURCE_DIR = Path("data/reference/fssai")
OUTPUT = "data/reference/fssai/fssai_pages.parquet"

SOURCE_PAGES = {
    "labelling_compendium.pdf":
        "https://fssai.gov.in/food-law/regulations",
    "labelling_amendment_2026.pdf":
        "https://www.fssai.gov.in/food-law/regulations/amendments/labelling-display"
}

files = sorted(SOURCE_DIR.glob("*.pdf"))

if len(files) != 2:
    raise SystemExit(
        f"Expected exactly 2 PDFs, found {len(files)}."
    )

records = []

for path in files:
    print("Reading:", path.name)

    document_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    reader = PdfReader(str(path))

    for page_number, page in enumerate(
        reader.pages, start=1
    ):
        error = ""

        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            error = str(exc)

        records.append({
            "document_name": path.name,
            "source_page_url": SOURCE_PAGES.get(
                path.name, ""
            ),
            "sha256": document_hash,
            "page_number": page_number,
            "page_text": text,
            "character_count": len(text),
            "word_count": len(
                re.findall(r"\b\w+\b", text)
            ),
            "extraction_error": error
        })

spark = (
    SparkSession.builder
    .appName("FSSAI_PDF_Integration")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.createDataFrame(records)

    df = df.withColumn(
        "has_text",
        F.col("character_count") > 0
    )

    print("\n=== FSSAI PDF EXTRACTION SUMMARY ===")

    df.groupBy("document_name").agg(
        F.count("*").alias("total_pages"),
        F.sum(
            F.col("has_text").cast("int")
        ).alias("pages_with_text"),
        F.sum("word_count").alias("total_words"),
        F.sum("character_count").alias(
            "total_characters"
        )
    ).show(truncate=False)

    print("\n=== REFERENCE TOPIC COVERAGE ===")

    for term in [
        "ingredient",
        "allergen",
        "nutrition",
        "additive",
        "label"
    ]:
        count = df.filter(
            F.lower(F.col("page_text")).contains(term)
        ).count()

        print(f"{term}: found on {count} pages")

    print("\n=== EXTRACTION QUALITY ===")

    df.agg(
        F.count("*").alias("pages_processed"),
        F.sum(
            (~F.col("has_text")).cast("int")
        ).alias("pages_without_text"),
        F.sum(
            (
                F.length("extraction_error") > 0
            ).cast("int")
        ).alias("pages_with_errors")
    ).show(truncate=False)

    # Save the extracted pages for later searching.
    df.write.mode("overwrite").parquet(OUTPUT)

    print("\nSaved:", OUTPUT)

finally:
    spark.stop()
