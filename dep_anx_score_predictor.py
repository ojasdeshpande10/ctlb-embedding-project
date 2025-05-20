
from pyspark.sql import SparkSession # type: ignore
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    TimestampType, ArrayType, DoubleType
)
from pyspark.sql.functions import size, col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType, ArrayType, DoubleType, FloatType
from pyspark.ml.feature import StandardScaler
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
import pickle
import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf

def getSecondLastLayerEmbeddings(df):

    def extract_last_list(values_list):
        return values_list[-2:-1]

    # Register it as a Spark UDF
    extract_udf = F.udf(extract_last_list, ArrayType(ArrayType(DoubleType())))

    # Apply this UDF to the DataFrame to get the last two lists
    lastLayerEmbeddings = df.withColumn("embedding_last_layer", extract_udf(F.col("embedding"))).drop(F.col("embedding"))

    def array_to_vector(arr):
        # Flatten the last two layers to create a single vector
        flat_arr = [item for sublist in arr for item in sublist]  # Flattening the list
        return Vectors.dense(flat_arr)

    # Register the UDF and apply it to convert the array to a vector
    array_to_vector_udf = F.udf(array_to_vector, VectorUDT())
    lastLayerEmbeddings = lastLayerEmbeddings.withColumn("embedding_last_layer_vector", array_to_vector_udf(lastLayerEmbeddings["embedding_last_layer"]))
    lastLayerEmbeddings = lastLayerEmbeddings.drop('embedding')
    lastLayerEmbeddings = lastLayerEmbeddings.drop('embedding_last_layer')
    return lastLayerEmbeddings

def read_in_pyspark(spark, path):
    # Define the schema
    schema = StructType([
        StructField("user_year_week", StringType(), True),
        StructField("user_id", LongType(), True),
        StructField("message_count", LongType(), True),
        StructField("embedding", ArrayType(
                DoubleType()
        ), True),
        StructField("fips", StringType(), True)
    ])

    # Read the Parquet file from HDFS with the defined schema
    df = spark.read.parquet(path)
    df.show()
    print("the number of rows is : ", df.count())
    valid_df = df.filter(size(df.embedding) == 4).filter(size(df.embedding[0]) == 1024)
    return valid_df


def generate_dep_score(last_two_layers_embeddings):

    def giveModel(model_file_path, score, scaler_attribute_name = "multiScalers", feature_name_attribute = "featureNamesList", model_type = "regressionModels"):
        model_data = pickle.load(open(model_file_path, "rb"))
        scaler = model_data[scaler_attribute_name][score][0]
        ridge_model = model_data[model_type][score]
        feature_cols =  model_data[feature_name_attribute][0]
        return ridge_model, scaler, feature_cols
    
    def adjust_embeddings(df, feature_names):
        def adjust_embedding(embedding):
            embedding_np = np.array(embedding)
            adjusted_embedding = np.zeros(len(feature_names))

            # Mapping features from the original embedding to the new adjusted embedding
            current_feature_indices = {f"{i}me": i for i in range(len(embedding_np))}
            for new_index, feature_name in enumerate(feature_names):
                if feature_name in current_feature_indices:
                    adjusted_embedding[new_index] = embedding_np[current_feature_indices[feature_name]]

            return adjusted_embedding.tolist()
        # Register the function as a UDF
        adjust_embedding_udf = F.udf(adjust_embedding, ArrayType(DoubleType()))
        # Apply the UDF to adjust the embeddings in the DataFrame
        adjusted_df = df.withColumn("adjusted_embedding", adjust_embedding_udf(F.col("embedding_last_layer_vector")))
        return adjusted_df
    
    def standardize_embeddings(df, scaler):
        # Define a function to scale the flattened embedding
        def scale_embedding(embedding):
            # Convert to numpy array and reshape for scaler
            embedding_np = np.array(embedding).reshape(1, -1)
            # Apply the scaler
            scaled_embedding = scaler.transform(embedding_np).flatten()
            # Convert back to list for Spark DataFrame compatibility
            return scaled_embedding.tolist()
        
        # Register the function as a UDF
        scale_embedding_udf = F.udf(scale_embedding, ArrayType(DoubleType()))
        # Apply the UDF to the embedding column and create a new DataFrame
        standardized_df = df.withColumn("standardized_embedding", scale_embedding_udf(F.col("adjusted_embedding")))
        return standardized_df

    # Load the models and scalers
    dep_ridge_model, dep_scaler, dep_feat_cols = giveModel(model_file_path="/home/odeshpande/models/dep_score.roberta_la_noSentL23.gft1000.pickle", score="dep_score")
    anx_ridge_model, anx_scaler, anx_feat_cols = giveModel(model_file_path="/home/odeshpande/models/anx_score.roberta_la_noSentL23.gft1000.pickle", score="anx_score")

    # Adjust the embeddings using the feature names
    dep_adjusted_df = adjust_embeddings(last_two_layers_embeddings, dep_feat_cols)
    anx_adjusted_df = adjust_embeddings(last_two_layers_embeddings, anx_feat_cols)
    # Standardize the embeddings using the loaded scalers
    dep_standardized_df = standardize_embeddings(dep_adjusted_df, dep_scaler)
    anx_standardized_df = standardize_embeddings(anx_adjusted_df, anx_scaler)
    
    def dep_model_predict(embeddings):
        embeddings = np.array(embeddings)
        # Assuming the model expects a 2D array for a single prediction
        prediction = dep_ridge_model.predict(embeddings.reshape(1, -1))
        return float(prediction[0])
    # Define the prediction function for the ANX model
    def anx_model_predict(embeddings):
        embeddings = np.array(embeddings)
        # Assuming the model expects a 2D array for a single prediction
        prediction = anx_ridge_model.predict(embeddings.reshape(1, -1))
        return float(prediction[0])
    

    # Register UDF for Spark DataFrame
    dep_predict_udf = F.udf(dep_model_predict, FloatType())
    anx_predict_udf = F.udf(anx_model_predict, FloatType())

    # Apply the prediction UDF to the DataFrame
    dep_df = dep_standardized_df.withColumn("dep_score", dep_predict_udf(col("standardized_embedding")))
    anx_df = anx_standardized_df.withColumn("anx_score", anx_predict_udf(col("standardized_embedding")))
    dep_df = dep_df.join(anx_df.select("user_year_week", "anx_score"), on="user_year_week", how = "inner")
    dep_df.show()
    dep_df = dep_df.drop('embedding_last_layer_vector')
    dep_df = dep_df.drop('standardized_embedding')
    dep_df = dep_df.drop('message_count')
    dep_df = dep_df.drop('adjusted_embedding')
    min_max_values_dep = dep_df.agg(
        F.min("dep_score").alias("min_score"),
        F.max("dep_score").alias("max_score")
    ).collect()[0]

    min_score_dep = min_max_values_dep["min_score"]
    max_score_dep = min_max_values_dep["max_score"]
    min_max_values_anx = dep_df.agg(
        F.min("anx_score").alias("min_score"),
        F.max("anx_score").alias("max_score")
    ).collect()[0]

    min_score_anx = min_max_values_anx["min_score"]
    max_score_anx = min_max_values_anx["max_score"]
    # Define a scaling function
    def scale_dep_score(df, min_score_dep, max_score_dep):
    # Scale dep_score between 0 and 5
        return df.withColumn(
            "scaled_dep_score",
            0 + 5 * (F.col("dep_score") - min_score_dep) / (max_score_dep - min_score_dep)
        )
    def scale_anx_score(df, min_score_anx, max_score_anx):
    # Scale anx_score between 0 and 5
        return df.withColumn(
            "scaled_anx_score",
            0 + 5 * (F.col("anx_score") - min_score_anx) / (max_score_anx - min_score_anx)
        )

    # Apply scaling
    scaled_df = scale_dep_score(dep_df, min_score_dep, max_score_dep)
    scaled_final_df = scale_anx_score(scaled_df, min_score_anx, max_score_anx)
    scaled_final_df.show()
    return scaled_final_df




def main():
    spark = SparkSession.builder \
    .appName("EmbeddingGroup") \
    .getOrCreate()
    embeddings_path_2020 = "hdfs://apollo-d0:9000/user/large-scale-embeddings/2020_usr-yr-week_embeddings/*.parquet"
    embeddings_path_2019 = "hdfs://apollo-d0:9000/user/large-scale-embeddings/2019_usr-yr-week_embeddings/*.parquet"
    df_2020 = read_in_pyspark(spark, embeddings_path_2020)
    df_2019 = read_in_pyspark(spark, embeddings_path_2019)
    

    df = df_2020.union(df_2019)
    print("the combined df has ", df.count())
    df.show()
    lastLayerEmbedding = getSecondLastLayerEmbeddings(df)
    lastLayerEmbedding.show()


    dep_score_df = generate_dep_score(lastLayerEmbedding)
    dep_score_df.coalesce(1).write.mode("append").csv('/user/large-scale-embeddings/2019-20_usr-yr-wk-cnty_scores', header=True)
    
if __name__ == "__main__":
    main()




# def extract_limit_scores(df):

#     sorted_df = df.orderBy('dep_score', ascending=True)
#     # Step 2: Extract the bottom 10 rows
#     bottom_10 = sorted_df.limit(10)

#     # Step 3: Extract the middle 10 rows
#     # First calculate the number of rows
#     total_rows = sorted_df.count()

#     # Find the middle 10 index range
#     middle_start = (total_rows // 2) - 5
#     middle_10 = sorted_df.limit(middle_start + 10).subtract(sorted_df.limit(middle_start))

#     # Step 4: Extract the top 10 rows (optional)
#     top_10 = sorted_df.orderBy('dep_score', ascending=False).limit(10)

#     # Step 5: Write each of them to separate CSV files
#     bottom_10.write.mode("overwrite").csv('/user/large-scale-embeddings/2020_dep-score_limits/top_10.csv', header=True)
#     middle_10.write.mode("overwrite").csv('/user/large-scale-embeddings/2020_dep-score_limits/middle_10.csv', header=True)
#     top_10.write.mode("overwrite").csv('/user/large-scale-embeddings/2020_dep-score_limits/bottom_10.csv', header=True)

