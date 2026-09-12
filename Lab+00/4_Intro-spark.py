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

userDir = "/Volumes/cscie103_catalog/lab_00/output/global/data"

# COMMAND ----------

# DBTITLE 1,Read the bike data into a Dataframe
filepath = '/databricks-datasets/definitive-guide/data/bike-data/201508_trip_data.csv'
print(filepath)
bikesDF = spark.read.option("header", True).option("inferSchema", True).csv(filepath)
bikesDF.printSchema()
display(bikesDF)

# COMMAND ----------

# DBTITLE 1,Accessing Columns
from pyspark.sql import functions as F

F.col("Start Date")
bikesDF["Start Terminal"]
bikesDF["Bike #"]

# COMMAND ----------

# DBTITLE 1,Column operations
F.col("End_Terminal") + F.col("Start_Terminal")
F.col("Start_Date").desc()
(F.col("Bike_#") * 100).cast("int")

# COMMAND ----------

# DBTITLE 1,Selecting Columns
subsetDF = bikesDF.select("Subscriber Type", "Zip Code")
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Selecting Columns with SQL expressions
subsetDF = bikesDF.selectExpr("Start Station", "`Bike #` between 1 and 100 as premium_bike")
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Dropping Columns
subsetDF = bikesDF.drop("user_id", "geo", "device")
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Adding a new Column
import random
subsetDF = bikesDF.withColumn("AdjDuration", F.round(F.col("Duration") + random.random() * random.uniform(5,9), 2))
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Filtering By Columns
subsetDF = bikesDF.filter("Start_Terminal > 50 and End_Terminal < 10")
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Sorting By Columns
import datetime
subsetDF = bikesDF.sort(F.col('End_date').desc(), 'Bike_#')
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Performing Aggregations
subsetDF1 = bikesDF.withColumn("Start_Date", F.date_format(F.to_date(F.col('Start_Date'), "M/d/yyyy H:m"), "MM/dd/yyyy"))
subsetDF2 = subsetDF1.withColumn("End_Date", F.date_format(F.to_date(F.col('End_Date'), "M/d/yyyy H:m"), "MM/dd/yyyy"))
subsetDF3 = subsetDF2.groupBy('Start_Date', 'End_Date').agg(F.sum("Duration").alias("AvgDuration")).sort('Start_Date','End_Date')
display(subsetDF3)

# COMMAND ----------

# DBTITLE 1,Another Dataframe with Maintenance Info
stationsDF = spark.read.format("parquet").load('/Volumes/cscie103_catalog/lab_00/output/global/data/bike-data-station/parquet/')
stationsDF=stationsDF.select('Maintenance_Station', 'Terminal')
display(stationsDF)

# COMMAND ----------

subsetDF

# COMMAND ----------

# DBTITLE 1,Define a Function
def convertToMins(secs):
  return round(secs/60, 3)


assert isinstance(convertToMins(100), float), "not a float"
convertToMins(100)

# COMMAND ----------

# DBTITLE 1,Define it as a User Defined Function (UDF) and use it within Dataframe operations
convertToMinsUDF = udf(convertToMins)
subsetDF = bikesDF.select('Trip_ID', convertToMinsUDF('Duration').alias('DurationMins'))
display(subsetDF)


# COMMAND ----------

subsetDF.explain()

# COMMAND ----------

# DBTITLE 1,Register Dataframe as a SQL Table

bikesDF.createOrReplaceTempView('bikesDFTable')


# COMMAND ----------

# DBTITLE 1,Use this temp table in SQL
# MAGIC %sql 
# MAGIC SELECT * FROM bikesDFTable WHERE to_date(Start_Date,'M/d/yyyy H:m') >= to_date('2015/08/01','yyyy/MM/dd') 
# MAGIC ORDER BY Start_Date

# COMMAND ----------

# DBTITLE 1,Output of SQL to a Dataframe
subsetDF = spark.sql('SELECT * from bikesDFTable where Start_Terminal=77 limit 10')
display(subsetDF)

# COMMAND ----------

# DBTITLE 1,Register a UDF to use in SQL
spark.udf.register("convertsqludf", convertToMins)

# COMMAND ----------

# DBTITLE 1,Use UDF in SQL
# MAGIC %sql
# MAGIC SELECT Trip_ID, convertsqludf(Duration) AS DurationMins, Start_Terminal 
# MAGIC FROM bikesDFTable

# COMMAND ----------

