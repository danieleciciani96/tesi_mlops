import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

#create version without iceberg extension options for CDE
spark = SparkSession.builder\
  .appName("0.2 - Batch Load into Icerberg Table") \
  .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-west-2")\
  .config("spark.kerberos.access.hadoopFileSystems", "s3a://ps-uat2")\
  .config("spark.jars","/home/cdsw/lib/iceberg-spark-runtime-3.2_2.12-0.13.2.jar")\
  .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
  .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
  .config("spark.sql.catalog.spark_catalog.type","hive") \
  .getOrCreate()

## Load data from Iceberg
df_raw = spark.sql("SELECT * FROM spark_catalog.default.pump_raw")

# NORMAL = 1 and RECOVERING & BROKEN = 0
df = df_raw.withColumn('machine_status_tmp', when(df_raw.machine_status == "NORMAL", 1)\
                                    .when(df_raw.machine_status == "RECOVERING", 0)\
                                    .when(df_raw.machine_status == "BROKEN", 0)\
                                    .otherwise('Unknown'))\
                                    .drop(df_raw.machine_status)\
                                    .withColumnRenamed("machine_status_tmp", "machine_status")

# Print target distributions
print("Distributions of target class: ")
print(df.groupBy('machine_status').count().orderBy('count').show())


# Fill na values with -1 
df = df.na.fill(value=-1)

# Select the relevant Feature
final_sensors = ['sensor_00','sensor_02', 'sensor_04', 'sensor_06',  'sensor_07', 'sensor_08',
                 'sensor_09', 'sensor_10', 'sensor_11', 'sensor_51', 'machine_status']


df = df.select(final_sensors)


try:
    df.writeTo("spark_catalog.default.pump_processed").using("iceberg").create()
    df.toPandas().to_csv('/home/cdsw/data/pump_processed.csv', index=False)

except:
    spark.sql("drop table spark_catalog.default.pump_processed")
    df.writeTo("spark_catalog.default.pump_processed").using("iceberg").create()
    df.toPandas().to_csv('/home/cdsw/data/pump_sensors_processed.csv', index=False)
    
spark.stop()


"""
# to write to s3
 df.write\
      .option("header","true")\
      .mode("overwrite")\
      .csv("s3a://ps-uat2/user/dciciani/pump_processed.csv")
"""
