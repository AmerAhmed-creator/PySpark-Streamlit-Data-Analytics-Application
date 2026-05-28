from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, trim, when, lower,
    split, regexp_extract
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

def create_spark_session():
    spark = SparkSession.builder \
        .appName("CarPricePipeline") \
        .getOrCreate()
    return spark


def load_data(spark, file_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def clean_data(df):
    # Remove " km" from mileage and convert to integer
    df = df.withColumn("Mileage", regexp_replace(col("Mileage"), " km", "")) \
           .withColumn("Mileage", col("Mileage").cast("int"))

    # Clean Engine volume (some have "Turbo", remove text)
    df = df.withColumn("Engine volume",
                       trim(regexp_replace(col("Engine volume"), "[A-Za-z ]", "")).cast("float"))

    # Clean Doors column (some have '04-May', keep only number)
    df = df.withColumn("Doors",
                       regexp_extract(col("Doors"), r"(\d+)", 1).cast("int"))

    # Convert Yes/No to boolean
    df = df.withColumn("Leather interior",
                       when(col("Leather interior") == "Yes", 1).otherwise(0))

    # Normalize manufacturer & model names
    df = df.withColumn("Manufacturer", trim(lower(col("Manufacturer"))))
    df = df.withColumn("Model", trim(lower(col("Model"))))

    # Remove rows with missing price
    df = df.filter(col("Price").isNotNull())

    return df


def build_ml_pipeline(df):
    categorical_cols = [
        "Manufacturer", "Model", "Category", "Fuel type",
        "Gear box type", "Drive wheels", "Wheel", "Color"
    ]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip")
                for c in categorical_cols]

    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
                for c in categorical_cols]

    numeric_cols = ["Engine volume", "Mileage", "Cylinders", "Doors", "Levy"]

    assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numeric_cols

    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_raw")

    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

    model = pipeline.fit(df)
    final_df = model.transform(df)

    return final_df


def save_output(df, output_path):
    df.select(
        "Price", "features"
    ).write.mode("overwrite").parquet(output_path)


def main():
    spark = create_spark_session()

    print("Loading dataset...")
    df = load_data(spark, "cars.csv")   # Change filename if needed

    print("Cleaning dataset...")
    df_clean = clean_data(df)

    print("Building ML pipeline...")
    df_final = build_ml_pipeline(df_clean)

    print("Saving processed dataset...")
    save_output(df_final, "processed_output")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
