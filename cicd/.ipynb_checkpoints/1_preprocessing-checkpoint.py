import numpy as np
import pandas as pd
import warnings
import sys
import findspark
findspark.init()
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

#create version without iceberg extension options for CDE
spark = SparkSession.builder\
  .appName("Prerocess pump_raw data") \
  .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-west-2")\
  .config("spark.kerberos.access.hadoopFileSystems", "s3a://ps-uat2")\
  .config("spark.jars","/home/cdsw/lib/iceberg-spark-runtime-3.2_2.12-0.13.2.jar")\
  .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
  .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
  .config("spark.sql.catalog.spark_catalog.type","hive") \
  .getOrCreate()

# get a snapshot_id from outside
snapshot_id = sys.argv[1] if len(sys.argv) > 1 else None


if len(sys.argv) > 1:
    df_raw = spark.read\
                  .option("snapshot-id", int(snapshot_id) )\
                  .table("spark_catalog.default.pump_raw")

## Else read the last version of the data
else:
    df_raw = spark.sql("SELECT * FROM spark_catalog.default.pump_raw")


# Adjust target class, NORMAL = 1 and RECOVERING & BROKEN = 0
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

# Select only the relevant Feature
final_sensors = ['timestamp', 'sensor_00','sensor_02', 'sensor_04', 'sensor_06',  'sensor_07', 'sensor_08',
                 'sensor_09', 'sensor_10', 'sensor_11', 'sensor_51', 'machine_status']


df = df.select(final_sensors)


try:
    df.writeTo("spark_catalog.default.pump_processed").using("iceberg").create()
    df.toPandas().to_csv('/home/cdsw/data/pump_processed.csv', index=False)

except:
    spark.sql("drop table spark_catalog.default.pump_processed")
    df.writeTo("spark_catalog.default.pump_processed").using("iceberg").create()
    df.toPandas().to_csv('/home/cdsw/data/pump_processed.csv', index=False)    
    
spark.stop()



"""
# to write to s3
 df.write\
      .option("header","true")\
      .mode("overwrite")\
      .csv("s3a://ps-uat2/user/dciciani/pump_processed.csv")
"""
