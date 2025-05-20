import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql.functions import col, year, count, concat, lit, date_format, to_date, weekofyear, sum, size
import pyspark.sql.functions as F
import time
from py4j.java_gateway import java_import
  
# def timer_func(func): 
#     # This function shows the execution time of  
#     # the function object passed 
#     def wrap_func(*args, **kwargs): 
#         t1 = time() 
#         result = func(*args, **kwargs) 
#         t2 = time() 
#         print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
#         return result 
#     return wrap_func

def filter_out_rts_urls(df):
    rt_regex = "^RT @\w+:"
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return df.filter(~F.col("message").rlike(rt_regex)) \
            .filter(~F.col("message").rlike(url_regex))

def fips_cleaner(df):
    fips_regex = "^\d{5}$"
    return df.filter(df.fips.rlike(fips_regex))
def fips_cleaner_not_5(df):
    return df.filter(~col("fips").cast("string").rlike(r"^\d{5}$"))

def normalize_user(df):
    df = df.withColumn("message", F.regexp_replace(df["message"], r"@\w+", "<USER>"))
    return df

def get_batch_number(spark, hdfs_dir):
    java_import(spark._jvm, "org.apache.hadoop.fs.FileSystem")
    java_import(spark._jvm, "org.apache.hadoop.fs.Path")

    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_dir)
    file_status_list = fs.listStatus(path)
    num_files = len(file_status_list)
    return num_files


def give_stats(df, emb):
    if emb:
        valid_df = df.filter(size(df.embedding) == 4).filter(size(df.embedding[0]) == 1024)
    else:
        valid_df = df
    distinct_count = valid_df.select("message_id").distinct().count()
    print("The number of messages are : ", distinct_count)
    print("the number of user-year-weeks :", valid_df.select("user_year_week").distinct().count())
    print("the number of users :", valid_df.select("user_id").distinct().count())
    print("The number of counties : ", valid_df.select("fips").distinct().count())

def main(input_path, existing_data_path, output_path):
    start_time = time.time()
    print("Starting Spark session creation")
    spark = SparkSession.builder \
        .appName("ParquetReadExample1") \
        .getOrCreate()
    print("Spark session created")

    columns = ['user_year_week', 'message_id', 'user_id', 'message', 'created_at', 'location', 'coordinates']
    spark.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")

    schema = StructType([
        StructField("user_year_week", StringType(), True),
        StructField("message_id", LongType(), True),
        StructField("user_id", LongType(), True),
        StructField("message", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("location", StringType(), True),
        StructField("coordinates", StringType(), True)
    ])

    df = spark \
        .read \
        .format("csv") \
        .option("header", False) \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .schema(schema) \
        .load(input_path) \
        .toDF(*columns)

    df = df.filter(df.user_id.isNotNull()) \
           .filter(df.message_id.isNotNull()) \
           .filter(df.created_at.isNotNull())
    df = normalize_user(df)
    df = df.withColumn("created_at", df["created_at"].cast("timestamp"))

    county_df = spark.read.format("csv").option("header", True).load("/user/hchoudhary/user_county_mapping.csv")
    result_df = df.join(county_df, "user_id", "inner")
    result_df = result_df.withColumnRenamed("cnty", "fips")
    result_df = filter_out_rts_urls(result_df)
    result_df = result_df.dropDuplicates(["message_id"])
    result_df = fips_cleaner_not_5(result_df)

    print("The number of rows sampled after dropping duplicates and filtering out retweets: ", result_df.count())
    end_time = time.time()
    print("Time taken to sample: ", end_time - start_time)

    message_counts_per_usr_yearweek = result_df.groupBy("user_year_week").agg(count("message_id").alias("message_count"))
    count_11_100 = message_counts_per_usr_yearweek.filter((col("message_count") >= 11) & (col("message_count") <= 100))
    count_101_above = message_counts_per_usr_yearweek.filter((col("message_count") >= 101))

    user_year_week_11_100 = count_11_100.select("user_year_week").distinct()
    user_year_week_101 = count_101_above.select("user_year_week").distinct()

    filtered_result_df1 = result_df.join(user_year_week_11_100, on="user_year_week", how="inner")
    # print("****User year week [11-100] messages*****")
    # give_stats(filtered_result_df1, False)
    filtered_result_df2 = result_df.join(user_year_week_101, on="user_year_week", how="inner")
    # print("****User year week [101-] messages*****")
    # give_stats(filtered_result_df2, False)
    superusers = filtered_result_df2.select("user_id").distinct()
    
    filtered_result_df1 = filtered_result_df1.join(superusers, on="user_id", how="left_anti")
    # print("****User year week [11-100] messages after removing superusers*****")
    # give_stats(filtered_result_df1, False)

    print("Messages in dataset after removing superusers: ", filtered_result_df1.count())

    schema = StructType([
        StructField("user_year_week", StringType(), True),
        StructField("user_id", LongType(), True),
        StructField("message_id", LongType(), True),
        StructField("message", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("location", StringType(), True),
        StructField("coordinates", StringType(), True),
        StructField("fips", StringType(), True)
    ])

    existing_data_frame = spark.read.schema(schema).parquet(existing_data_path)
    filtered_result_df1 = filtered_result_df1.join(existing_data_frame, on="message_id", how="left_anti")

    print("Messages in dataset after removing existing messages: ", filtered_result_df1.count())
    batches = get_batch_number(spark, output_path)
    print("Existing number of batches: ", batches)

    msg_counts = filtered_result_df1.groupBy("user_year_week").agg(count("message_id").alias("message_count"))
    msg_count_sorted = msg_counts.orderBy(col("message_count"))
    rows = msg_count_sorted.collect()

    message_limit = 3000000
    current_sum = 0
    current_batch = []
    total_sum = 0
    for row in rows:
        message_count = row["message_count"]
        if current_sum + message_count > message_limit:
            print("filtering the batch")
            batch_df = filtered_result_df1.filter(F.col("user_year_week").isin([r["user_year_week"] for r in current_batch]))
            batch_df.repartition(200).write.format("parquet").mode("append").save(output_path + '2019_sampled_data_11_100_usr_year_week_batch' + str(batches) + '/')
            print("batch printed")
            print("Current_sum : ", current_sum)
            batches += 1
            current_sum = 0
            current_batch = []

        current_sum += message_count
        current_batch.append(row)
        if batches > 60: 
            break

    if current_batch:
        batch_df = filtered_result_df1.filter(F.col("user_year_week").isin([r["user_year_week"] for r in current_batch]))
        batch_df.write.format("parquet").mode("append").save(output_path + '2019_sampled_data_11_100_usr_year_week_batch' + str(batches) + '/')
        print(f"Written final batch number {batches}")

    print("Batches created for 2019 data: ", batches)
    print("The number of counties: ", filtered_result_df1.select("fips").distinct().count())
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output paths for the Spark job.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--existing_data_path", type=str, required=True, help="Path to the existing data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    main(args.input_path, args.existing_data_path, args.output_path)

'''

existing-path ("hdfs://apollo-d0:9000/user/large-scale-embeddings/sampled_data_2020/2020_sampled_data_11_100_usr_year_week_batch*/*.parquet")
output_path = '/user/large-scale-embeddings/sampled_data_2020/2020_sampled_data_11_100_usr_year_week_batch'

     batches = get_batch_number(spark, "hdfs://apollo-d0:9000/user/large-scale-embeddings/sampled_data_2020/")
    # Define the path to your CSV file
    path = "/apollo_data/ctlb/2020/feats/timelines2020_full_3upts.csv"



'''