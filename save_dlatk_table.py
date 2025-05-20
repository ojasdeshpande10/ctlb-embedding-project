from pyspark.sql.functions import col, flatten, posexplode, concat_ws, lit, row_number, lpad
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def save_as_dlatk_feat_table(final_df, column_name):
    embed_avg = final_df.select("fips", posexplode(col(column_name)).alias("index", "value"))
    embed_avg = embed_avg.withColumn("feat", concat_ws("", col("index").cast("string"), lit("me")))
    embed_avg = embed_avg.withColumn("group_norm", col("value"))

    embed_avg = embed_avg.drop('index')
    embed_avg = embed_avg.withColumnRenamed("fips", "group_id")
    global_window = Window.orderBy(lit(1))
    embed_avg = embed_avg.withColumn("id", row_number().over(global_window))
    cols = embed_avg.columns
    new_order = ["id"] + [col for col in cols if col != "id"]
    embed_avg = embed_avg.select(new_order)
    embed_avg.show()
    return embed_avg
def read_missing_county_embeddings(spark):
    missing_county_embeddings_path = "/user/large-scale-embeddings/missing-county-2020/county_embeddings_2020/*.parquet"
    missing_county_embeddings_df = spark.read.parquet(missing_county_embeddings_path)
    print(missing_county_embeddings_df.count())
    duplicate_fips_count = (
    missing_county_embeddings_df.groupBy("fips")
      .count()
      .filter("count > 1")
      .agg(F.sum("count") - F.count("fips"))
      .first()[0]
    )
    print(f"Number of duplicate rows based on fips: {duplicate_fips_count}")
    dedup_df = missing_county_embeddings_df.dropDuplicates(["fips"])
    print(dedup_df.count())
    dedup_df.show()
    df = dedup_df.withColumn("fips", lpad(col("fips").cast("string"), 5, "0"))
    return df
def main():
    spark = SparkSession.builder \
        .appName("Save as dlatk feat table") \
        .getOrCreate()

    # missing county embeddings
    missing_county_embeddings_df = read_missing_county_embeddings(spark)
    missing_county_embeddings_df.show()
    vanilla_embeddings_df = spark.read.parquet("/user/large-scale-embeddings/ctlb_2020_cnty_embedding/*.parquet")
    print(vanilla_embeddings_df.select("fips").distinct().count())
    vanilla_embeddings_df.show()

    county_2020_df = vanilla_embeddings_df.union(missing_county_embeddings_df)
    print(county_2020_df.select('fips').distinct().count())
    # cnty_embedding = save_as_dlatk_feat_table(county_2020_df, "embedding")
    cnty_wavg_embedding = save_as_dlatk_feat_table(county_2020_df, "weighted_embedding")
    # cnty_embedding.coalesce(1).write.format("csv").mode("append").save('/user/large-scale-embeddings/')
    cnty_wavg_embedding.coalesce(1).write.format("csv").mode("append").save('/user/large-scale-embeddings/')

    # save_as_dlatk_feat_table(county_2020_df, "embedding")

if __name__ == "__main__":
    main()
