# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md #Housing Prices Prediction Application
# MAGIC This is an end-to-end example of using machine learning algorithms to solve a supervised regression problem.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ###Table of Contents
# MAGIC
# MAGIC - *Step 1: Business Understanding*
# MAGIC - *Step 2: Load Your Data*
# MAGIC - *Step 3: Explore Your Data*
# MAGIC - *Step 4: Visualize Your Data*
# MAGIC - *Step 5: Data Preparation*
# MAGIC - *Step 6: Data Modeling*
# MAGIC - *Step 7: Evaluation*
# MAGIC
# MAGIC
# MAGIC
# MAGIC *We are trying to predict housing prices given a set of prices from various zipcodes.  The dataset is rich in features and includes number of bedrooms, bathrooms, sq footage area etc..*
# MAGIC
# MAGIC
# MAGIC Given this business problem, we need to translate it to a Machine Learning task.  The ML task is regression since the label (or target) we are trying to predict is numeric.
# MAGIC
# MAGIC
# MAGIC The example data is available at https://raw.githubusercontent.com/grzegorzgajda/spark-examples/master/spark-examples/data/house-data.csv
# MAGIC
# MAGIC
# MAGIC More information about Machine Learning with Spark can be found in the [Spark ML Programming Guide](https://spark.apache.org/docs/latest/ml-guide.html)

# COMMAND ----------

# %run
# %sql
# %py
# %sh
# %md

# COMMAND ----------

# This cell is idempotent.
spark.sql('CREATE CATALOG IF NOT EXISTS cscie103_catalog');
spark.sql('USE CATALOG cscie103_catalog')

spark.sql('CREATE SCHEMA IF NOT EXISTS lab_01');
spark.sql('USE cscie103_catalog.lab_01')

spark.sql('CREATE VOLUME IF NOT EXISTS input')
spark.sql('CREATE VOLUME IF NOT EXISTS output')

# COMMAND ----------

# MAGIC %sh 
# MAGIC wget https://raw.githubusercontent.com/grzegorzgajda/spark-examples/master/spark-examples/data/house-data.csv -P /Volumes/cscie103_catalog/lab_01/input

# COMMAND ----------

dbutils.fs.cp("/Volumes/cscie103_catalog/lab_01/input/house-data.csv", "/Volumes/cscie103_catalog/lab_01/output/housing/house-data.csv", True)

# COMMAND ----------

userDir = "/Volumes/cscie103_catalog/lab_01"

# COMMAND ----------

dbutils.fs.ls(f"{userDir}/output/housing/")

# COMMAND ----------

# MAGIC %md ##Step 1: Business Understanding
# MAGIC The first step in any machine learning task is to understand the business need. 
# MAGIC
# MAGIC As described in the overview we are trying to predict housing price given a set of prices from various house types in the Seattle Metro area.
# MAGIC
# MAGIC The problem is a regression problem since the label (or target) we are trying to predict is numeric

# COMMAND ----------

# MAGIC %md ##Step 2: Load Your Data
# MAGIC Now that we understand what we are trying to do, we need to load our data and describe it, explore it and verify it.

# COMMAND ----------

from pyspark.sql.functions import *

data = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(f"{userDir}/output/housing/"))

display(data)

# COMMAND ----------

dataDF = (data
        .withColumn('date', from_unixtime(unix_timestamp(col('date'), 'yyyyMMdd\'T\'HHmmss')))
        .withColumn("zipcode", data["zipcode"].cast("string")))

# COMMAND ----------

dataDF.printSchema()

# COMMAND ----------

display(dataDF)

# COMMAND ----------

dataDF.write.format("delta").mode("overwrite").saveAsTable("tblHousing")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tblHousing;

# COMMAND ----------

# MAGIC %md ##Step 3: Explore Your Data
# MAGIC Now that we understand what we are trying to do, we need to load our data and describe it, explore it and verify it.

# COMMAND ----------

# MAGIC %md We can use the SQL desc command to describe the schema

# COMMAND ----------

# MAGIC %sql 
# MAGIC desc tblhousing;

# COMMAND ----------

# DBTITLE 1,Get Pricing Trends by Zip code
# MAGIC %sql select zipcode, month(date) as month, avg(price) from tblhousing where month(date) in (1,2,3) group by zipcode, month order by zipcode,month

# COMMAND ----------

# MAGIC %md ##Step 4: Visualize Your Data
# MAGIC
# MAGIC To understand our data, we will look for correlations between features and the label.  This can be important when choosing a model.  E.g., if features and a label are linearly correlated, a linear model like Linear Regression can do well; if the relationship is very non-linear, more complex models such as Decision Trees can be better. We use the Databricks built in visualization to view each of our predictors in relation to the label column as a scatter plot to see the correlation between the predictors and the label.

# COMMAND ----------

# DBTITLE 1,Check for correlations between price and living area
# MAGIC %sql select * from tblhousing

# COMMAND ----------

# MAGIC %sql select avg(condition) as condition, bedrooms, bathrooms from tblhousing group by bedrooms, bathrooms having condition <> 3 order by bedrooms, bathrooms asc

# COMMAND ----------

# MAGIC %md ##Step 5: Data Preparation
# MAGIC
# MAGIC The next step is to prepare the data. Since all of this data is numeric and consistent, this is a simple task for us today.
# MAGIC
# MAGIC We will need to convert the predictor features from columns to Feature Vectors using the org.apache.spark.ml.feature.VectorAssembler
# MAGIC
# MAGIC The VectorAssembler will be the first step in building our ML pipeline.

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

stringIndexer = StringIndexer(inputCol="zipcode", outputCol="zipcodeIndex")
assembler = VectorAssembler(
    inputCols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built', 'waterfront', 'zipcodeIndex', 'condition'],
    outputCol = "features")

pipeline = Pipeline(stages=[stringIndexer, assembler])
outputDF = pipeline.fit(dataDF).transform(dataDF)
display(outputDF)

# COMMAND ----------

# MAGIC %md ##Step 6: Data Modeling
# MAGIC Now let's model our data to predict what the power output will be given a set of sensor readings
# MAGIC
# MAGIC Our first model will be based on simple linear regression since we saw some linear patterns in our data based on the scatter plots during the exploration stage.

# COMMAND ----------

(trainDF,testDF) = outputDF.randomSplit([0.8,0.2], seed = 123)

# COMMAND ----------

# MAGIC %md The cell below is based on the Spark ML pipeline API. More information can be found in the Spark ML Programming Guide at https://spark.apache.org/docs/latest/ml-guide.html

# COMMAND ----------

# MAGIC %md 
# MAGIC Since Linear Regression is simply a line of best fit over the data that minimizes the square of the error, given multiple input dimensions we can express each predictor as a line function of the form:
# MAGIC
# MAGIC \\(y = a + b x_1 + b x_2 + b x_i ...  \\)
# MAGIC
# MAGIC where a is the intercept and b are coefficients.
# MAGIC
# MAGIC To express the coefficients of that line we can retrieve the Estimator stage from the PipelineModel and express the weights and the intercept for the function.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

niter = 500         # set by trial and error
reg = 0.05          # no regularization for now
elastic_reg = 0.05  # secondary regularization parameter (ratio of L1 to L2 regularization penalties)

lr = LinearRegression(featuresCol="features", labelCol="price", 
                      maxIter=niter, regParam=reg, 
                      elasticNetParam=elastic_reg)

# Fit the model
lrModel = lr.fit(trainDF)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
#print("numIterations: %d" % trainingSummary.totalIterations)
#print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
#trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

# MAGIC %md 
# MAGIC Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit
# MAGIC
# MAGIC
# MAGIC Now let's see what our predictions look like given this model.

# COMMAND ----------

#get original zipcodes and create a sql table which we can use to run queries
predictions = lrModel.transform(testDF)
display(predictions.select('price', 'prediction', 'bedrooms', 'bathrooms', 'zipcode', 'condition', 'sqft_lot', 'yr_built'))

# COMMAND ----------

# %sql
# -- Optional cell if you run into errors in the overwrite in the next cell
# drop table tblPredictions

# COMMAND ----------

predictions.write.format("delta").mode('overwrite').saveAsTable('tblPredictions')

# COMMAND ----------

# MAGIC %sql select price, prediction, bedrooms, bathrooms, zipcode, condition, sqft_lot, yr_built from tblPredictions

# COMMAND ----------

# MAGIC %md #Step 7: Evaluation
# MAGIC
# MAGIC Lets analyze the model.

# COMMAND ----------

query = "select price, prediction as predictedPrice, (price-prediction) as Residual_Error, ((price-prediction)/%s) as Within_RMSE from tblPredictions" % lrModel.summary.rootMeanSquaredError
rmseEvaluator = spark.sql(query)
rmseEvaluator.createOrReplaceTempView("RMSE_Evaluation")

# COMMAND ----------

# MAGIC %sql select * from RMSE_Evaluation

# COMMAND ----------

# MAGIC %sql -- Now we can display the RMSE as a Histogram. Clearly this shows that the RMSE is centered around 0 with the vast majority of the error within 2 RMSEs.
# MAGIC SELECT Within_RMSE from RMSE_Evaluation

# COMMAND ----------

# MAGIC %md Let's see within what price range do we have an accurate prediction.

# COMMAND ----------

# MAGIC %md We can see this definitively if we count the number of predictions within + or - 1.0 and + or - 2.0 and display this as a pie chart:

# COMMAND ----------

# MAGIC %sql SELECT 
# MAGIC         CASE WHEN Within_RMSE <=1.0 AND Within_RMSE >= -1.0 THEN 1 
# MAGIC              WHEN Within_RMSE <=2.0 AND Within_RMSE >= -2.0 THEN 2 
# MAGIC              ELSE 3
# MAGIC         END RMSE_Multiple, 
# MAGIC         count(*) count 
# MAGIC     from RMSE_Evaluation group by RMSE_Multiple

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC We have
# MAGIC ####81% of our test data is within RMSE 1 and 96% within RMSE 2.#### 