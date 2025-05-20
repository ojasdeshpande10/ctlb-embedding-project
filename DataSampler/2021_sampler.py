
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType
from pyspark.sql.functions import col, year, count, concat, lit, date_format, to_date, weekofyear, sum
import pyspark.sql.functions as F

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

    df.show()
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

    # Using user to county id mapping to map users to heir locations

    county_df = spark.read.format("csv").option("header", True).load("/user/hchoudhary/user_county_mapping.csv")

    result_df = df.join(county_df, "user_id", "inner")

    print("The number of rows of sampled data matching with county data: ", result_df.count()) 

    result_df.show()
    
    # result_df = result_df.withColumnRenamed("cnty", "fips")
    # result_df = filter_out_rts_urls(result_df)
    # result_df = result_df.filter(F.col("message_id").isNotNull())
    # print("LOG: Filtered Null values from the messages")

    # result_df = result_df.dropDuplicates(["message_id"])
    # print("LOG: Duplicates dropped from Tweets")

    # result_df = fips_cleaner(result_df)
    # print("the number of rows sampled after dropping duplicates and filtering out rewteets : ", result_df.count())
    # ### Storing the dataset at user-year-week level 
    # result_df = result_df.withColumn("user_year_week", concat(col("user_id"), lit(":"), col("year"), lit("-"), col("week")))


    # print("Number of messages in 2021 data : ", result_df.select("message_id").distinct().count())
    # print("Number of users in 2021 data : ", result_df.select("user_id").distinct().count())
    # print("Number of counties in 2021 data : ", result_df.select("fips").distinct().count())
    # print("Number of user-year-weeks 2021  data : ", result_df.select("user_year_week").distinct().count())

    # # Counting messages per user year week
    # message_counts_per_usr_yearweek = result_df.groupBy("user_year_week").agg(count("message_id").alias("message_count"))
    # # Keeping only those user-year-week where the total of message_counts is more than 11
    # output_path = '/user/large-scale-embeddings/user-year-week_mapping_2021/'
    # # count_11_above = message_counts_per_usr_yearweek.filter((col("message_count") >= 11))
    # # count_1_10 = message_counts_per_usr_yearweek.filter((col("message_count") >= 1) & (col("message_count") <= 10))
    # # count_11_100 = message_counts_per_usr_yearweek.filter((col("message_count") >= 11) & (col("message_count") <= 100))
    # count_101_above = message_counts_per_usr_yearweek.filter((col("message_count") >= 101))


    # # user_year_week_11_100 = count_11_100.select("user_year_week").distinct()
    # user_year_week_101 = count_101_above.select("user_year_week").distinct()
    # # user_year_week_1_10 = count_1_10.select("user_year_week").distinct()

    # # print("Number of user-year-week having 11-100 messages : ", count_11_100.select("user_year_week").distinct().count())
    # # total_messages_11_100 = count_11_100.agg(sum("message_count").alias("TotalMessages"))
    # # print("Number of messages in 11-100 category")
    # # total_messages_11_100.show()


    # print("Number of user-year-week having 101 messages : ", count_101_above.select("user_year_week").distinct().count())
    # total_messages_101 = count_101_above.agg(sum("message_count").alias("TotalMessages"))
    # print("Number of messages in 100+ category")
    # total_messages_101.show()


    # # print("Number of user-year-week having 101 messages : ", count_1_10.select("user_year_week").distinct().count())
    # # total_messages_1_10 = count_1_10.agg(sum("message_count").alias("TotalMessages"))
    # # print("Number of messages in 1-10 category")
    # # total_messages_1_10.show()

    # # Filtering the original dataframe on the filter on users having more than 10 messages per week
    # # filtered_result_df1 = result_df.join(user_year_week_11_100, on="user_year_week", how="inner")
    # filtered_result_df2 = result_df.join(user_year_week_101, on="user_year_week", how="inner")
    # # filtered_result_df3 = result_df.join(user_year_week_1_10, on="user_year_week", how="inner")

    # # print("Number of users in 11-100 : ", filtered_result_df1.select("user_id").distinct().count())
    # print("Number of users in 101+ : ", filtered_result_df2.select("user_id").distinct().count())  
    # # print("Number of users in 1-10 : ", filtered_result_df3.select("user_id").distinct().count()) 

    # superusers_df =  filtered_result_df2.select("user_id").distinct()

    # superusers_df.coalesce(1).write.format("parquet").mode("append").save('/user/large-scale-embeddings/')


    # print("Number of counties in this 11-100 group :", filtered_result_df1.select("fips").distinct().count())
    # print("Number of couties in this 101+ group :", filtered_result_df2.select("fips").distinct().count())
    # print("Number of couties in this 1-10 group :", filtered_result_df3.select("fips").distinct().count())

    spark.stop()
if __name__ == "__main__":
    main()
 