
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql.functions import col, year, count, concat, lit, date_format, to_date, weekofyear, sum
import pyspark.sql.functions as F
from pyspark.sql import Row

def filter_out_rts_urls(df):
    rt_regex = "^RT @\w+:"
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return df.filter(~F.col("message").rlike(rt_regex)) \
            .filter(~F.col("message").rlike(url_regex))


def fips_cleaner(df):
    fips_regex = "^\d{5}$"
    return df.filter(df.fips.rlike(fips_regex))

def normalize_user(df):

    df = df.withColumn("message", F.regexp_replace(df["message"], r"@\w+", "<USER>"))
    return df


def main():

    # Initialize Spark Session
    spark = SparkSession.builder.appName("CSVReaderWithSchema").getOrCreate()
    columns = ['message_id', 'user_id', 'message', 'created_at', 'location', 'coordinates']

    # Define the schema with specific data types for the columns

    # Define the path to your CSV file
    path = "/apollo_data/ctlb/2021/*.csv"

    schema = StructType([
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

    df.printSchema()

    # Filter DataFrame to keep only rows where the 'user_id', 'message_id', 'created-at' column is not null
    df = df.filter(df.user_id.isNotNull())
    df = df.filter(df.message_id.isNotNull())
    df = df.filter(df.created_at.isNotNull())
    # df2 = filtered_df.filter(filtered_df.location.isNotNull())

    df = normalize_user(df)


    df = df.withColumn("created_at", df["created_at"].cast("timestamp"))
    # Creating new column with only date
    df = df.withColumn("date_only", to_date(col("created_at")))
    # year column
    df = df.withColumn("year", year(df["date_only"]))
    df = df.filter(df["year"] == 2021)
    # Week column
    df = df.withColumn("week", weekofyear(df["date_only"]))


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

    result_df = fips_cleaner(result_df)
    print("the number of rows sampled after dropping duplicates and filtering out rewteets : ", result_df.count())
    ### Storing the dataset at user-year-week level 
    result_df = result_df.withColumn("user_year_week", concat(col("user_id"), lit(":"), col("year"), lit("-"), col("week")))

    # Counting messages per user year week
    message_counts_per_usr_yearweek = result_df.groupBy("user_year_week").agg(count("message_id").alias("message_count"))
    # Keeping only those user-year-week where the total of message_counts is more than 11
    output_path = '/user/large-scale-embeddings/user-year-week_mapping_2021/'
    # count_11_above = message_counts_per_usr_yearweek.filter((col("message_count") >= 11))
    count_11_100 = message_counts_per_usr_yearweek.filter((col("message_count") >= 11) & (col("message_count") <= 100))
    result_df = result_df.drop("date_only")
    result_df = result_df.drop("year")
    result_df = result_df.drop("week")
    msg_count_sorted = count_11_100.orderBy(col("message_count"))

    rows = msg_count_sorted.collect()

    message_limit = 5000000
    current_sum = 0
    batch_number = 0
    batches = []
    output_path = '/user/large-scale-embeddings/sampled_data_2021/2021_sampled_data_11_100_usr_year_week_batch'

    # Loop through the rows and assign batches
    for row in rows:
        message_count = row["message_count"]
        if current_sum + message_count > message_limit:
            df_current_batch = spark.createDataFrame(batches).select("user_year_week")
            joint_temp_df = result_df.join(df_current_batch, on = "user_year_week", how = "inner")
            print("the number of messages in this batch are : ",joint_temp_df.count())
            joint_temp_df.write.format("parquet").mode("append").save(output_path+str(batch_number)+'/')
            print("Written batch-number : ", batch_number)
            batch_number += 1
            current_sum = 0
            batches=[]
        
        current_sum += message_count
        batches.append(Row(user_year_week=row["user_year_week"], message_count=row["message_count"], batch=batch_number))


    print(batch_number)
    spark.stop()
if __name__ == "__main__":
    main()
 