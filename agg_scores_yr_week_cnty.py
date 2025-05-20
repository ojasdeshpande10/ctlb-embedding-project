import csv, argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from tqdm import tqdm
from pyspark.sql.functions import when, col, split, concat, substring_index, coalesce, lit, count, avg, stddev_pop, countDistinct


DEF_INPUTFILE = "/hadoop_data/ctlb/2019/feats/featANS.dd_daa_c2adpt_ans_nos.timelines2019_full_3upts.yw_user_id.cnty"
DEF_MAPPING_FILE_1 = "/user/large-scale-embeddings/user_wts19to20/income_weights_redist_5_50.csv"
DEF_MAPPING_FILE_2 = "/user/large-scale-embeddings/user_wts19to20/income_weights_redist_50_10.csv"

def create_spark_session():
    return SparkSession.builder \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .appName("addWeightToFeats") \
        .getOrCreate()
def main():
    # Parse arguments

    # Create SparkSession
    session = create_spark_session()
    mapping = session.read.csv( "/user/large-scale-embeddings/user_wts19to20/income_weights_redist_5_50.csv", header=True, inferSchema=True)
    dep_anx_df = session.read.csv("/user/large-scale-embeddings/2019-20_usr-yr-wk-cnty_scores/rb_la_noSent_scores.csv", header=True, inferSchema=True)
    dep_anx_df.show()
    print("the dep anx df has ", dep_anx_df.count())

    df_with_weight1 = dep_anx_df.join(mapping, on="user_id", how="left_outer")
    df_with_weight1 = df_with_weight1.withColumn("weight", coalesce(df_with_weight1["weight"], lit(1.0)))
    df_with_weight1.show()
    print("the joint df1 has ", df_with_weight1.count())

    null_cnty_count_before = df_with_weight1.filter(col("cnty").isNull()).count()
    print("Number of rows with null cnty before modification:", null_cnty_count_before)

    # Update cnty column to be equal to fips when cnty is NULL
    df_with_weight1 = df_with_weight1.withColumn("cnty", when(df_with_weight1["cnty"].isNull(), df_with_weight1["fips"]).otherwise(df_with_weight1["cnty"]))

    # Count null cnty values after modification
    null_cnty_count_after = df_with_weight1.filter(col("cnty").isNull()).count()
    print("Number of rows with null cnty after modification:", null_cnty_count_after)

    df_with_weight1.show()
    print("the joint df1 has ", df_with_weight1.count())

    different_fips_cnty_count = df_with_weight1.filter(col("fips") != col("cnty")).count()
    print("Number of rows where fips and cnty are different:", different_fips_cnty_count)

    # Show some examples of rows where fips and cnty are different
    print("Examples of rows where fips and cnty are different:")
    df_with_weight1.filter(col("fips") != col("cnty")).show(5)

    df_with_weight1 = df_with_weight1.drop('cnty')
    print("the joint df1 has ", df_with_weight1.count())

    df_with_weight1 = df_with_weight1.withColumnRenamed('weight', 'user_weight')
    df_with_weight1 = df_with_weight1.withColumn(
    "year_week_county",
    concat(
        substring_index(col("user_year_week"), ":", 1),  # Extract everything before ':'
        lit("_"),  # Add underscore separator
        col("fips")  # Add county
        )
    )

    dep_score_df = df_with_weight1.select(
        "user_id", "year_week_county", "fips", "user_weight",
        F.lit("DEP_SCORE").alias("feat"),
        F.col("scaled_dep_score").alias("feat_score")
        
    )

    dep_score_df.show()

    anx_score_df = df_with_weight1.select(
        "user_id", "year_week_county", "fips", "user_weight",
        F.lit("ANX_SCORE").alias("feat"),
        F.col("scaled_anx_score").alias("feat_score")
        
    )

    anx_score_df.show()
    
    transformed_df = dep_score_df.union(anx_score_df)
    transformed_df.show()
    weighted_averages = transformed_df.groupBy("year_week_county", "feat").agg(
        (F.sum(F.col("feat_score") * F.col("user_weight")) / F.sum(F.col("user_weight"))).alias("wavg_score"),
        avg(F.col("feat_score") ).alias('avg_score'),
        # ((F.sum(col("user_weight")*(col("feat_score") - col("wavg_score"))**2)/F.sum(col("user_weight")))**0.5).alias('std_score_wt'),
        stddev_pop(F.col("feat_score") ).alias('std_score'),
        countDistinct(F.col("user_id")).alias('N_users')
    )

    # TODO add stddev weighted ((F.sum(col(yearweek_userid_weight)*(col(field_freq_field) - col("avg_score_wt"))**2)/F.sum(col(yearweek_userid_weight)))**0.5).alias('std_score_wt'),

    print("Dataframe with weighted averages by year_week_county:")
    weighted_averages.show(10, truncate=False)

    # Get the total number of groups
    total_groups = weighted_averages.count()
    print(f"Total number of year_week_county groups: {total_groups}")
    # Display some statistics about the weighted averages

    weighted_averages.coalesce(1).write.mode("append").csv('/user/large-scale-embeddings/2019-20_yr-wk-cnty_scores/', header=True)

    session.stop()



if __name__ == "__main__":
    main()
