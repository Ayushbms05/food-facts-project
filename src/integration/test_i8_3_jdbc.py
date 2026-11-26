"""I8.3.3: small PostgreSQL -> Spark -> PostgreSQL round trip."""
from pathlib import Path
import re
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[2]
JDBC_DRIVER = "org.postgresql:postgresql:42.7.14"
URL = "jdbc:postgresql://127.0.0.1:5432/foodfacts"
SOURCE = "demo.products"
TARGET = "demo.spark_products"


def main():
    # Read the password created in the previous step without printing it.
    env = (ROOT / "infra" / ".env").read_text()
    passwords = re.findall(r"^POSTGRES_PASSWORD=([0-9a-f]{48})$", env, re.M)
    if len(passwords) != 1:
        raise ValueError("Expected the generated hex password in infra/.env; do not share it.")
    props = {"user": "food_admin", "password": passwords[0],
             "driver": "org.postgresql.Driver"}
    spark = (SparkSession.builder.appName("FoodFactsJDBCTest")
             .master("local[2]")
             .config("spark.driver.memory", "1g")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.jars.packages", JDBC_DRIVER)
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        print("Spark version:", spark.version)
        print("JDBC dependency:", JDBC_DRIVER)
        original = spark.read.jdbc(URL, SOURCE, properties=props)
        print("EXTRACT: rows from PostgreSQL")
        original.orderBy("gtin14").show(truncate=False)
        expected = [
            ("00000000000017", "Milk", "Milk", 1),
            ("00000000000024", "Cooking oil", "Sunflower oil", 2),
            ("00000000000031", "Biscuits", "Flour, sugar, flavouring", 4),
        ]
        fields = ["gtin14", "product_name", "ingredients", "nova_group"]
        assert [tuple(r) for r in original.select(*fields).orderBy("gtin14").collect()] == expected, "Demo source records do not match the previous exercise"
        transformed = original.select(*fields).withColumn("product_name_upper", F.upper("product_name"))
        print("TRANSFORM: add uppercase name")
        transformed.orderBy("gtin14").show(truncate=False)
        print("LOAD: writing new demo.spark_products table (no overwrite)")
        transformed.coalesce(1).write.jdbc(URL, TARGET, mode="errorifexists", properties={**props, "batchsize": "2"})
        returned = spark.read.jdbc(URL, TARGET, properties=props)
        print("VERIFY: rows read back from PostgreSQL")
        returned.orderBy("gtin14").show(truncate=False)
        actual = [tuple(r) for r in returned.select(*fields, "product_name_upper").orderBy("gtin14").collect()]
        assert actual == [(*row, row[1].upper()) for row in expected], "Round-trip values differ"
        assert [tuple(r) for r in spark.read.jdbc(URL, SOURCE, properties=props).select(*fields).orderBy("gtin14").collect()] == expected, "Source table changed"
        print("SUCCESS: 3/3 rows and all values verified; original demo table unchanged.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
