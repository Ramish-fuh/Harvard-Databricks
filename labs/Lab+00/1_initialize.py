# Databricks notebook source
# MAGIC %md
# MAGIC ### You can run this notebook 
# MAGIC ### It is an initialization notebook that writes out some datasets to use in other labs

# COMMAND ----------

# This cell is idempotent.
spark.sql('CREATE CATALOG IF NOT EXISTS cscie103_catalog');
spark.sql('USE CATALOG cscie103_catalog')

spark.sql('CREATE SCHEMA IF NOT EXISTS lab_00');
spark.sql('USE cscie103_catalog.lab_00')

spark.sql('CREATE VOLUME IF NOT EXISTS input')
spark.sql('CREATE VOLUME IF NOT EXISTS output')

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/definitive-guide/data/bike-data

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/definitive-guide/data/bike-data/201508_trip_data.csv

# COMMAND ----------

# MAGIC %fs rm -r /Volumes/cscie103_catalog/lab_00/output

# COMMAND ----------

usersCsvPath = "/databricks-datasets/definitive-guide/data/bike-data/201508_trip_data.csv"

usersDF = (spark.read
  .option("sep", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv(usersCsvPath))

new_column_name_list= list(map(lambda x: x.replace(" ", "_"), usersDF.columns))
usersDF = usersDF.toDF(*new_column_name_list)

usersDF.printSchema()

# COMMAND ----------

(usersDF
  .coalesce(1)
  .write
  .option("sep", ",")
  .option("header", True)
  .mode('overwrite')
  .format("csv")
  .save('/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/csv2/'))

# COMMAND ----------

usersDF.coalesce(1).write.mode('overwrite').json("/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/json/")

# COMMAND ----------

usersDF.coalesce(1).write.mode('overwrite').parquet("/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data/parquet/")

# COMMAND ----------

spark.read.format("csv").load("dbfs:/databricks-datasets/definitive-guide/data/bike-data/201508_station_data.csv", header=True).selectExpr("station_id as Terminal", "name as Maintenance_Station").write.format("parquet").save("/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data-station/parquet/")

# COMMAND ----------

usersDF.createOrReplaceTempView('a')
display(usersDF)