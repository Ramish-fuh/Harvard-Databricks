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
# MAGIC # Databricks Platform
# MAGIC 1. Execute code in multiple languages
# MAGIC 1. Create documentation cells
# MAGIC 1. Access DBFS (Databricks File System)
# MAGIC 1. Create database and table
# MAGIC 1. Query table and plot results
# MAGIC
# MAGIC ##### Databricks Notebook Utilities
# MAGIC - <a href="https://docs.databricks.com/notebooks/notebooks-use.html#language-magic" target="_blank">Magic commands</a>: `%python`, `%scala`, `%sql`, `%r`, `%sh`, `%md`
# MAGIC - <a href="https://docs.databricks.com/dev-tools/databricks-utils.html" target="_blank">Databricks utilities</a>: `dbutils.fs` (`%fs`), `dbutils.notebooks` (`%run`), `dbutils.widgets`
# MAGIC - <a href="https://docs.databricks.com/notebooks/visualizations/index.html" target="_blank">Visualization</a>: `display`, `displayHTML`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute code in multiple languages
# MAGIC Run default language of notebook

# COMMAND ----------

# MAGIC %md
# MAGIC Run language specified by language magic command: `%python`, `%scala`, `%sql`, `%r`

# COMMAND ----------

print("Runnin' python")

# COMMAND ----------

# MAGIC %sql
# MAGIC select "Runnin' SQL"

# COMMAND ----------

# MAGIC %md
# MAGIC Run shell commands using magic command: `%sh`

# COMMAND ----------

# MAGIC %sh ps -eaf | tr -s ' ' | cut -d ' ' -f 8

# COMMAND ----------

# MAGIC %md
# MAGIC Render HTML using the function: `displayHTML` (available in Python, Scala, and R)

# COMMAND ----------

html = """
	<body style="background-color: #f5f5f5; color: #008080;">
    <h1 style="background-color: #008080; color: #f5f5f5;">Spark Overview</h1>
    <p>Apache Spark is a unified analytics engine for large-scale data processing</p>
</body>"""
displayHTML(html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use cells for Documentation
# MAGIC Render cell as <a href="https://www.markdownguide.org/cheat-sheet/" target="_blank">Markdown</a> using magic command: `%md`
# MAGIC
# MAGIC Double click on cell to see markdown
# MAGIC
# MAGIC # Spark Overview
# MAGIC ### Markdown
# MAGIC > a block quote
# MAGIC
# MAGIC 1. **spark (bold)**
# MAGIC 2. *dataframes (italicized)*
# MAGIC 3. ~~data siloes (strikethrough)~~
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Access DBFS (Databricks File System)
# MAGIC Run file system commands on DBFS using magic command: `%fs`

# COMMAND ----------

# DBTITLE 1,dbfs root
# MAGIC %fs ls

# COMMAND ----------

# DBTITLE 1,Lots of sample datasets
# MAGIC %fs ls /databricks-datasets

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/samples/people

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/cs100/lab4/data-001/

# COMMAND ----------

# DBTITLE 1,Display top few lines in file
# MAGIC %fs head /databricks-datasets/README.md

# COMMAND ----------

# MAGIC %md
# MAGIC `%fs` is shorthand for the <a href="https://docs.databricks.com/dev-tools/databricks-utils.html" target="_blank">DBUtils</a> module: `dbutils.fs`

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

# MAGIC %fs help

# COMMAND ----------

# MAGIC %md
# MAGIC Run file system commands on DBFS using DBUtils directly

# COMMAND ----------

dbutils.fs.ls("/databricks-datasets")

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize results in a table using the Databricks <a href="https://docs.databricks.com/notebooks/visualizations/index.html#display-function-1" target="_blank">display</a> function

# COMMAND ----------

files = dbutils.fs.ls("/databricks-datasets")
display(files)

# COMMAND ----------

dbutils.fs.head('/databricks-datasets/samples/lending_club/readme.md')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create table
# MAGIC Run <a href="https://docs.databricks.com/spark/latest/spark-sql/language-manual/index.html#sql-reference" target="_blank">Databricks SQL Commands</a> to create a table named `events` .

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS tbllendingclub
# MAGIC AS
# MAGIC SELECT *
# MAGIC FROM parquet.`/databricks-datasets/samples/lending_club/parquet/`;

# COMMAND ----------

# MAGIC %md
# MAGIC This table was saved in the database created for you in classroom setup. Database name is printed below. 

# COMMAND ----------

# MAGIC %md
# MAGIC View your database and table in the Data tab of the UI. Check this out in the left ribbon bar

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query table and plot results
# MAGIC Use SQL to query `tbllendingclub` table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM tbllendingclub LIMIT 100;

# COMMAND ----------

# MAGIC %md
# MAGIC Run the query below and then <a href="https://docs.databricks.com/notebooks/visualizations/index.html#plot-types" target="_blank">plot</a> results by selecting the bar chart icon

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT zip_code, loan_status, avg(loan_amnt) AS avg_loan, avg(total_pymnt) AS avg_paid 
# MAGIC FROM tbllendingclub 
# MAGIC WHERE INSTR(zip_code, "xx") >= 1 --valid zip code?
# MAGIC GROUP BY zip_code, loan_status
# MAGIC ORDER BY zip_code, loan_status

# COMMAND ----------

