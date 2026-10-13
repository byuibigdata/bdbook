from pyspark.sql import SparkSession, functions as F
from pyspark.ml.fpm import FPGrowth

spark = (SparkSession.builder.master("local[4]")
         .config("spark.driver.memory", "6g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

items = spark.read.parquet("pdi_cstore_transection_items/*.parquet")
gtin = spark.read.parquet("pdi_one/cstore_master_ctin.parquet").select("GTIN", "CATEGORY")

silver = (items
    .filter((F.col("UNIT_QUANTITY") > 0) & (F.col("GRAND_TOTAL_AMOUNT") > 0))
    .join(gtin, on="GTIN", how="left")
    .withColumn("category", F.coalesce("CATEGORY", "NONSCAN_CATEGORY"))
    .filter(F.col("category").isNotNull()))

baskets = (silver.groupBy("TRANSACTION_SET_ID")
    .agg(F.collect_set("category").alias("items"))
    .withColumn("n", F.size("items")))
total = baskets.count()
multi = baskets.filter(F.col("n") > 1)
print("baskets", total, "multi-item", multi.count())

model = FPGrowth(itemsCol="items", minSupport=0.005, minConfidence=0.05).fit(multi)
rules = (model.associationRules
    .withColumn("antecedent", F.concat_ws(" + ", "antecedent"))
    .withColumn("consequent", F.concat_ws(" + ", "consequent"))
    .orderBy(F.desc("lift")))
rules.select("antecedent", "consequent",
             F.round("confidence", 3).alias("confidence"),
             F.round("lift", 2).alias("lift"),
             F.round("support", 4).alias("support")).show(15, truncate=False)

freq = model.freqItemsets.withColumn("n", F.size("items")).filter("n = 1") \
    .withColumn("item", F.concat_ws("", "items")) \
    .withColumn("support", F.round(F.col("freq") / multi.count(), 3))
freq.select("item", "freq", "support").orderBy(F.desc("freq")).show(10, truncate=False)

# supervised target: did the basket include fuel, and how much was spent on it
fuel = (silver.groupBy("TRANSACTION_SET_ID")
    .agg(F.max(F.when(F.col("category") == "FUEL", 1).otherwise(0)).alias("bought_fuel"),
         F.sum(F.when(F.col("category") == "FUEL", F.col("GRAND_TOTAL_AMOUNT")).otherwise(0.0)).alias("fuel_spend")))
fuel.agg(F.round(F.avg("bought_fuel"), 3).alias("fuel_rate"),
         F.round(F.avg(F.when(F.col("bought_fuel") == 1, F.col("fuel_spend"))), 2).alias("avg_fuel_spend")).show()
