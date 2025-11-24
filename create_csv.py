# This file is used to create CSV files from the parquet files of the monthly dataset

import numpy as np
import pandas as pd
import glob

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("test") \
    .master("local[*]") \
    .getOrCreate()

# Follows 3 Steps
# 1. Get a list of all parquet files using Python
file_list = glob.glob("2025/*.parquet")

# 2. Pass the list to Spark
df = spark.read.parquet(*file_list)

# 3. Makes sure that the output is a single CSV file
df.coalesce(1).write.csv('NYC_Yellow_Taxi_2025_data_till_August.csv')

# Now, for the previous year's files
file_list = glob.glob("2024/*.parquet")

df = spark.read.parquet(*file_list)

df.coalesce(1).write.csv('NYC_Yellow_Taxi_2024_data.csv')
