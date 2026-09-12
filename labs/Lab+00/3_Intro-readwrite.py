# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# This cell is idempotent.
spark.sql('CREATE CATALOG IF NOT EXISTS cscie103_catalog');
spark.sql('USE CATALOG cscie103_catalog')

spark.sql('CREATE SCHEMA IF NOT EXISTS lab_00');
spark.sql('USE cscie103_catalog.lab_00')

spark.sql('CREATE VOLUME IF NOT EXISTS input')
spark.sql('CREATE VOLUME IF NOT EXISTS output')

# COMMAND ----------

# MAGIC %md
# MAGIC # Reader & Writer
# MAGIC 1. Read from CSV files
# MAGIC 1. Read from JSON files
# MAGIC 1. Write DataFrame to files
# MAGIC 1. Write DataFrame to tables
# MAGIC
# MAGIC ##### Methods
# MAGIC - DataFrameReader (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=dataframereader#pyspark.sql.DataFrameReader" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/DataFrameReader.html" target="_blank">Scala</a>): `csv`, `json`, `option`, `schema`
# MAGIC - DataFrameWriter (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=dataframereader#pyspark.sql.DataFrameWriter" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/DataFrameWriter.html" target="_blank">Scala</a>): `mode`, `option`, `parquet`, `format`, `saveAsTable`
# MAGIC - StructType (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=structtype#pyspark.sql.types.StructType" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/types/StructType.html" target="_blank" target="_blank">Scala</a>): `toDDL`
# MAGIC
# MAGIC ##### Spark Types
# MAGIC - Types (<a href="https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=types#module-pyspark.sql.types" target="_blank">Python</a>/<a href="http://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/types/index.html" target="_blank">Scala</a>): `ArrayType`, `DoubleType`, `IntegerType`, `LongType`, `StringType`, `StructType`, `StructField`

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Read from CSV files
# MAGIC Read from CSV with DataFrameReader's `csv` method and the following options:
# MAGIC
# MAGIC Tab separator, use first line as header, infer schema

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets

# COMMAND ----------

# MAGIC %fs ls /Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/csv2/

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/definitive-guide/data/bike-data

# COMMAND ----------

# MAGIC %fs cp /databricks-datasets/definitive-guide/data/bike-data/201508_trip_data.csv 

# COMMAND ----------

usersCsvPath = "/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/csv2/"

usersDF = (spark.read
  .option("sep", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv(usersCsvPath))

usersDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Manually define the schema by creating a `StructType` with column names and data types

# COMMAND ----------

from pyspark.sql.types import IntegerType, StringType, StructType, StructField

userDefinedSchema = StructType([
  StructField("Trip_ID", IntegerType(), True),  
  StructField("Duration", IntegerType(), True),
  StructField("Start_Date", StringType(), True),
  StructField("Start_Station", StringType(), True),
  StructField("Start_Terminal", IntegerType(), True),
  StructField("End_Date", StringType(), True),
  StructField("End_Station", StringType(), True),
  StructField("End_Terminal", IntegerType(), True),
  StructField("Bike_#", IntegerType(), True),
  StructField("Subscriber_Type", StringType(), True),
  StructField("Zip_Code", IntegerType(), True), #change to Integer Type
  StructField("Maintenance", IntegerType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC Read from CSV using this user-defined schema instead of inferring schema

# COMMAND ----------

usersDF = (spark.read
  .option("sep", ",")
  .option("header", True)
  .schema(userDefinedSchema)
  .csv(usersCsvPath))

# COMMAND ----------

display(usersDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, define the schema using a DDL formatted string.

# COMMAND ----------

DDLSchema = """Trip_ID INT,Duration INT,Start_Date string,
Start_Station string,Start_Terminal INT,
End_Date string,End_Station string,End_Terminal INT,
`Bike_#` INT,Subscriber_Type string, Zip_Code INT, Maintenance INT"""

usersDF = (spark.read
  .option("sep", ",")
  .option("header", True)
  .schema(DDLSchema)
  .csv(usersCsvPath))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read from JSON files
# MAGIC
# MAGIC Read from JSON with DataFrameReader's `json` method and the infer schema option

# COMMAND ----------

usersJsonPath = "/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/json/"

usersDF = (spark.read
  .option("inferSchema", True)
  .json(usersJsonPath))

display(usersDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Read data faster by creating a `StructType` with the schema names and data types

# COMMAND ----------

usersDF = (spark.read
  .schema(userDefinedSchema)
  .json(usersJsonPath))

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Write DataFrames to files
# MAGIC
# MAGIC Write `usersDF` to parquet with DataFrameWriter's `parquet` method and the following configurations:
# MAGIC
# MAGIC Snappy compression, overwrite mode

# COMMAND ----------

# MAGIC %fs mkdirs "/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/parquet/"
# MAGIC
# MAGIC

# COMMAND ----------


usersOutputPath = "/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/parquet/"

(usersDF.write
  .option("compression", "snappy")
  .mode("overwrite")
  .parquet(usersOutputPath)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Write `usersDF` to <a href="https://delta.io/" target="_blank">Delta</a> with DataFrameWriter's `save` method and the following configurations:
# MAGIC
# MAGIC Delta format, overwrite mode

# COMMAND ----------

usersOutputPath = "/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/delta/"

(usersDF.write
  .format("delta")
  .mode("overwrite")
  .save(usersOutputPath)
)

# COMMAND ----------

# DBTITLE 1,Note the presence of the _delta_log folder. This is where the Delta transaction logs are stored
dbutils.fs.ls(usersOutputPath)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Write DataFrames to tables
# MAGIC
# MAGIC Write `usersDF` to a table using the DataFrameWriter method `saveAsTable`
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This creates a persistent global table, unlike the local view created by the DataFrame method `createOrReplaceTempView`

# COMMAND ----------

usersDF.write.mode("overwrite").saveAsTable("tblBikeData")

# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

# MAGIC %md
# MAGIC This table was saved in the database created for you in classroom setup. See database name printed below. Also check the `data` widget in the ribbon on the left

# COMMAND ----------

# MAGIC %sql
# MAGIC select current_database()

# COMMAND ----------

