import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import *


#create sparkSession with iceberg extension

spark = SparkSession.builder\
  .appName("Refresh Raw into Icerberg Table") \
  .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-west-2")\
  .config("spark.kerberos.access.hadoopFileSystems", "s3a://ps-uat2")\
  .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
  .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
  .config("spark.sql.catalog.spark_catalog.type","hive") \
  .getOrCreate()



#read raw data 
df_raw = spark.sql("SELECT * FROM spark_catalog.default.pump_raw")


print("Total row count in the raw table before batch load")
print(df_raw.count())

# Add a sample in the table to simule a batch ingestion 10% of row data
df_raw_new = df_raw.sample(withReplacement=True, fraction=0.1)

print("Delta rows to add")
print(df_raw_new.count())


# Write the sample in a stagin table
df_raw_new.writeTo("spark_catalog.default.pump_staging").using("iceberg").create()
# Insert the sample in the pump_raw table
spark.sql("INSERT INTO spark_catalog.default.pump_raw SELECT * FROM spark_catalog.default.pump_staging").show()
# Drop the staging table
spark.sql("DROP TABLE IF EXISTS spark_catalog.default.pump_staging")

print("Total row count in the target table after batch load")
print(spark.sql("SELECT * FROM spark_catalog.default.pump_raw").count())


spark.stop()