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

# DBTITLE 1,Set up catalog, schema and volume
# This cell is idempotent.
spark.sql('CREATE CATALOG IF NOT EXISTS cscie103_catalog');
spark.sql('USE CATALOG cscie103_catalog')

spark.sql('CREATE SCHEMA IF NOT EXISTS lab_01');
spark.sql('USE cscie103_catalog.lab_01')

spark.sql('CREATE VOLUME IF NOT EXISTS input')
spark.sql('CREATE VOLUME IF NOT EXISTS output')

# COMMAND ----------

# DBTITLE 1,Shell command to download source file. store in volume
# MAGIC %sh 
# MAGIC wget https://raw.githubusercontent.com/grzegorzgajda/spark-examples/master/spark-examples/data/house-data.csv -P /Volumes/cscie103_catalog/lab_01/input

# COMMAND ----------

# DBTITLE 1,Illustrative: Filesystem command to copy across paths
dbutils.fs.cp("/Volumes/cscie103_catalog/lab_01/input/house-data.csv", "/Volumes/cscie103_catalog/lab_01/output/housing/house-data.csv", True)

# COMMAND ----------

# DBTITLE 1,Set userDir for convenience later
userDir = "/Volumes/cscie103_catalog/lab_01"

# COMMAND ----------

# DBTITLE 1,List files to verify
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

# DBTITLE 1,Read the CSV data into a dataframe
from pyspark.sql.functions import *

data = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(f"{userDir}/output/housing/"))

display(data)

# COMMAND ----------

# DBTITLE 1,Basic Transformations
dataDF = (data
        .withColumn('date', from_unixtime(unix_timestamp(col('date'), 'yyyyMMdd\'T\'HHmmss')))
        .withColumn("zipcode", data["zipcode"].cast("string")))

# COMMAND ----------

display(dataDF)

# COMMAND ----------

# DBTITLE 1,Write into a Delta Table
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
# MAGIC -- desc tblhousing;
# MAGIC describe extended tblhousing;

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

# MAGIC %md 
# MAGIC ##Step 5: Data Preparation
# MAGIC
# MAGIC The next step is to prepare the data. 
# MAGIC Since all of this data is numeric and consistent, this is a simple task for us today.
# MAGIC
# MAGIC ### Orig SparkML (other Notebook does not work on Serverless yet)
# MAGIC We will need to convert the predictor features from columns to Feature Vectors using the org.apache.spark.ml.feature.VectorAssembler
# MAGIC The VectorAssembler will be the first step in building our ML pipeline.
# MAGIC
# MAGIC ### This Notebook - uses pandas and scikit-learn (LabelEncoder)
# MAGIC
# MAGIC

# COMMAND ----------

spark.conf.set("spark.sql.ansi.enabled", "false")

# COMMAND ----------

# DBTITLE 1,Define Features
import pyspark.pandas as ps
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Convert Spark DataFrame to pandas-on-Spark DataFrame
pdf = dataDF.toPandas()

# Encode 'zipcode' using LabelEncoder
le = LabelEncoder()
pdf['zipcodeIndex'] = le.fit_transform(pdf['zipcode'])

# Assemble features into a numpy array
feature_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                   'yr_built', 'waterfront', 'zipcodeIndex', 'condition']
pdf['features'] = pdf[feature_columns].values.tolist()

# Convert back to pyspark.pandas DataFrame if needed for downstream compatibility
outputDF = ps.from_pandas(pdf)
# display(outputDF)
spark_df = outputDF.to_spark()
display(spark_df)

# COMMAND ----------

# MAGIC %md ##Step 6: Data Modeling
# MAGIC Now let's model our data to predict what the power output will be given a set of sensor readings
# MAGIC
# MAGIC Our first model will be based on simple linear regression since we saw some linear patterns in our data based on the scatter plots during the exploration stage.

# COMMAND ----------

# DBTITLE 1,Split into Train and Test dataframes
(trainDF,testDF) = spark_df.randomSplit([0.8,0.2], seed = 123)

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

# DBTITLE 1,Train LR model and evaluate performance
import pyspark.pandas as ps
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Convert trainDF to pandas-on-Spark DataFrame
train_pdf = trainDF.toPandas()

# Extract features and labels
X_train = np.array(train_pdf['features'].tolist())
y_train = train_pdf['price'].values

# Fit the linear regression model using scikit-learn
lr = LinearRegression()
lr.fit(X_train, y_train)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lr.coef_))
print("Intercept: %s" % str(lr.intercept_))

# Calculate metrics on the training set
predictions = lr.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train, predictions))
r2 = r2_score(y_train, predictions)

print("RMSE: %f" % rmse)
print("r2: %f" % r2)

# COMMAND ----------

# MAGIC %md 
# MAGIC Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit
# MAGIC
# MAGIC
# MAGIC Now let's see what our predictions look like given this model.

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Using scikit-learn and pyspark.pandas (single-node ML)
# MAGIC The other notebook uses SparkML's lrModel.transform and writes predictions to a Delta table. To align with the single-node ML approach, we should use the predictions from scikit-learn (from Cell 29), combine them with the test set, and display the relevant columns. We will use pyspark.pandas for compatibility and avoid any SparkML or distributed ML code.

# COMMAND ----------

# DBTITLE 1,Process Test Data. Make predictions.
import pyspark.pandas as ps
import pandas as pd

# Convert testDF to pandas-on-Spark DataFrame
test_pdf = testDF.toPandas()

# Extract features and labels
X_test = np.array(test_pdf['features'].tolist())
y_test = test_pdf['price'].values

# Make predictions using the trained scikit-learn model
predictions = lr.predict(X_test)

# Add predictions to the test DataFrame
test_pdf['prediction'] = predictions

# Convert back to pyspark.pandas DataFrame for compatibility
predictions_psdf = ps.from_pandas(test_pdf)

# Display the selected columns
display(predictions_psdf[['price', 'prediction', 'bedrooms', 'bathrooms', 'zipcode', 'condition', 'sqft_lot', 'yr_built']])

# If you want to save predictions to a Delta table, convert to Spark DataFrame and write
predictions_sdf = predictions_psdf.to_spark()

# COMMAND ----------

# %sql
# -- Optional cell if you run into errors in the overwrite in the next cell
# drop table tblPredictions

# COMMAND ----------

# DBTITLE 1,Write to Delta Table
predictions_sdf.write.format("delta").mode('overwrite').option('overwriteSchema', 'true').saveAsTable('tblPredictions')

# COMMAND ----------

# DBTITLE 1,Compare Price and Prediction
# MAGIC %sql select price, prediction, bedrooms, bathrooms, zipcode, condition, sqft_lot, yr_built from tblPredictions

# COMMAND ----------

# MAGIC %md #Step 7: Evaluation
# MAGIC
# MAGIC Lets analyze the model.

# COMMAND ----------

# DBTITLE 1,RMSE
# Calculate RMSE using scikit-learn or your model's predictions
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_value = np.sqrt(
    mean_squared_error(
        test_pdf['price'],
        test_pdf['prediction']
    )
)

# Register the predictions DataFrame as a temp view if not already done
spark_df = spark.createDataFrame(test_pdf)
spark_df.createOrReplaceTempView("tblPredictions")

# Use the calculated RMSE in your SQL query
query = f"""
select
    price,
    prediction as predictedPrice,
    (price - prediction) as Residual_Error,
    ((price - prediction) / {rmse_value}) as Within_RMSE
from tblPredictions
"""
rmseEvaluator = spark.sql(query)
rmseEvaluator.createOrReplaceTempView("RMSE_Evaluation")

# COMMAND ----------

# MAGIC %sql select * from RMSE_Evaluation

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Now we can display the RMSE as a Histogram. Clearly this shows that the RMSE is centered around 0 with the vast majority of the error within 2 RMSEs.
# MAGIC SELECT Within_RMSE from RMSE_Evaluation

# COMMAND ----------

# MAGIC %md Let's see within what price range do we have an accurate prediction.

# COMMAND ----------

# MAGIC %md We can see this definitively if we count the number of predictions within + or - 1.0 and + or - 2.0 and display this as a pie chart:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CASE
# MAGIC     WHEN
# MAGIC       Within_RMSE <= 1.0 AND Within_RMSE >= - 1.0
# MAGIC     THEN
# MAGIC       1
# MAGIC     WHEN
# MAGIC       Within_RMSE <= 2.0 AND Within_RMSE >= - 2.0
# MAGIC     THEN
# MAGIC       2
# MAGIC     ELSE 3
# MAGIC   END RMSE_Multiple,
# MAGIC   count(*) count
# MAGIC from
# MAGIC   RMSE_Evaluation
# MAGIC group by
# MAGIC   RMSE_Multiple

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC We have
# MAGIC ####81% of our test data is within RMSE 1 and 96% within RMSE 2.#### 