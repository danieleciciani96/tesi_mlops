import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *


spark = SparkSession.builder\
  .appName("0.2 - Batch Load into Icerberg Table") \
  .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-west-2")\
  .config("spark.kerberos.access.hadoopFileSystems", "s3a://ps-uat2")\
  .config("spark.jars","/home/cdsw/lib/iceberg-spark-runtime-3.2_2.12-0.13.2.jar") \
  .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
  .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
  .config("spark.sql.catalog.spark_catalog.type","hive") \
  .getOrCreate()


## Load data to Iceberg
df = spark.read.format('csv').options(header='true', inferSchema='true').load('s3a://ps-uat2/user/dciciani/pump_sensor.csv')

spark.sql("DROP TABLE IF EXISTS spark_catalog.default.pump_raw")
df.writeTo("spark_catalog.default.pump_raw").using("iceberg").create()

spark.stop()