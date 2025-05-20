
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql.functions import col, year, count, concat, lit, date_format, to_date, weekofyear, sum
import pyspark.sql.functions as F
import time

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
def fips_cleaner_1_4(df):
    fips_regex = r"^\d{1,4}$"
    return df.filter(df.fips.rlike(fips_regex))


def normalize_user(df):

    df = df.withColumn("message", F.regexp_replace(df["message"], r"@\w+", "<USER>"))
    return df


def main():

    # Initialize Spark Session

    start_time = time.time()
    print("starting spark creation")
    spark = SparkSession.builder \
    .appName("ParquetReadExample1") \
    .getOrCreate()
    print("spark session created")
    columns = ['user_year_week','message_id', 'user_id', 'message', 'created_at', 'location', 'coordinates']

    # Define the schema with specific data types for the columns

    # Define the path to your CSV file
    path = "/apollo_data/ctlb/2020/feats/timelines2020_full_3upts.csv"

    schema = StructType([
        StructField("user_year_week", StringType(), True),
        StructField("message_id", LongType(), True),  # BigInt equivalent
        StructField("user_id", LongType(), True),  # BigInt equivalent
        StructField("message", StringType(), True),
        StructField("created_at", StringType(), True),  # Timestamp format
        StructField("location", StringType(), True),
        StructField("coordinates", StringType(), True)
    ])

    # Read the CSV file into a DataFrame with the defined schema and handle malformed rows
    df = spark \
        .read \
        .format("csv") \
        .option("header", False) \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .schema(schema) \
        .load(path) \
        .toDF(*columns)

    # Filter DataFrame to keep only rows where the 'user_id', 'message_id', 'created-at' column is not null
    df = df.filter(df.user_id.isNotNull())
    df = df.filter(df.message_id.isNotNull())
    df = df.filter(df.created_at.isNotNull())
    # df2 = filtered_df.filter(filtered_df.location.isNotNull())
    print("the number of users : ", df.select("user_id").distinct().count())
    print("the number of messages : ", df.select("message_id").distinct().count())

    df = normalize_user(df)
    df = df.withColumn("created_at", df["created_at"].cast("timestamp"))


    print("The number of rows before the joining with county data: ", df.count())

    # Using user to county id mapping to map users to their locations

    county_df = spark.read.format("csv").option("header", True).load("/user/hchoudhary/user_county_mapping.csv")

    result_df = df.join(county_df, "user_id", "inner")

    print("The number of rows of sampled data matching with county data: ", result_df.count()) 
    
    result_df = result_df.withColumnRenamed("cnty", "fips")
    result_df = filter_out_rts_urls(result_df)
    result_df = result_df.filter(F.col("message_id").isNotNull())
    print("LOG: Filtered Null values from the messages")

    result_df = result_df.dropDuplicates(["message_id"])
    print("LOG: Duplicates dropped from Tweets")

    result_df = fips_cleaner_not_5(result_df)
    print("the number of rows sampled after dropping duplicates and filtering out rewteets : ", result_df.count())
    end_time=time.time()

    print("the time taken to sample is ", end_time-start_time)
    ### Storing the dataset at user-year-week level 

    # Counting messages per user year week
    message_counts_per_usr_yearweek = result_df.groupBy("user_year_week").agg(count("message_id").alias("message_count"))
    # Keeping only those user-year-week where the total of message_counts is more than 11
    output_path = '/user/large-scale-embeddings/usr-yr-wk_mapping_2020/'
    # count_11_above = message_counts_per_usr_yearweek.filter((col("message_count") >= 11))
    count_11_100 = message_counts_per_usr_yearweek.filter((col("message_count") >= 11) & (col("message_count") <= 100))
    count_101_above = message_counts_per_usr_yearweek.filter((col("message_count") >= 101))
    # count_11_100.coalesce(1) \
    # .write \
    # .format("csv") \
    # .option("header", "true") \
    # .option("delimiter", ",") \
    # .mode("append") \
    # .save(output_path)

    # user_year_weeks_11_above = count_11_above.select("user_year_week").distinct()
    # print("Number of user-year-week in the dataset ", user_year_weeks_11_above.count())

    user_year_week_11_100 = count_11_100.select("user_year_week").distinct()
    user_year_week_101 = count_101_above.select("user_year_week").distinct()

    print("The number of user year weeks in 11-100 messages group : ",user_year_week_11_100.count())
    print("the number of user year weeks in 101+ messages group : ",user_year_week_101.count())


    # Filtering the original dataframe on the filter on users having more than 10 messages per week
    filtered_result_df1 = result_df.join(user_year_week_11_100, on="user_year_week", how="inner")
    filtered_result_df2 = result_df.join(user_year_week_101, on="user_year_week", how="inner")

    print("Number of users in 11-100 : ", filtered_result_df1.select("user_id").distinct().count())
    print("Number of users in 101+ : ", filtered_result_df2.select("user_id").distinct().count())
    superusers = filtered_result_df2.select("user_id").distinct()
    superusers.show()
    print("the number of messages before removing superusers : ", filtered_result_df1.select("message_id").distinct().count())
    print("the number of users before removing superusers : ", filtered_result_df1.select("user_id").distinct().count())
    print("the number of counties before removing superusers : ", filtered_result_df1.select("fips").distinct().count())
    filtered_result_df1 = filtered_result_df1.join(superusers, on="user_id",how="left_anti")   
    print("the number of messages after removing superusers : ", filtered_result_df1.select("message_id").distinct().count())
    print("the number of users after removing superusers : ", filtered_result_df1.select("user_id").distinct().count())
    print("the number of counties after removing superusers : ", filtered_result_df1.select("fips").distinct().count())
    print("the number of user-year-week after removing superusers : ", filtered_result_df1.select("user_year_week").distinct().count())
    spark.stop()
if __name__ == "__main__":
    main()
 