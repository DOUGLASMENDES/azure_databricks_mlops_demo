# Databricks notebook source
# MAGIC %run ../config_env

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Model
# MAGIC 
# MAGIC After prepared the dataset and load features on feature table, we'll training our machine learning model.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot
%matplotlib inline
import seaborn as sns

import collections
from mlflow.tracking import MlflowClient

from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Load Data

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

# Get today date
retrieve_date = datetime.date.today()

# Using today date for time travel
#customer_features_df = fs.read_table(name=feature_store_db_name_and_table, as_of_delta_timestamp=str(retrieve_date))
customer_features_df = fs.read_table(name=feature_store_db_name_and_table)

# COMMAND ----------

display(customer_features_df)

# COMMAND ----------

corr = customer_features_df.toPandas().drop(columns=['CustomerID']).corr()
corr.style.background_gradient(cmap='Blues_r')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 3. Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

SEED = 2022

customer_features_pd = customer_features_df.toPandas()
# Target column
target_col = "Churn"
# Features
split_X = customer_features_pd.drop([target_col, 'CustomerID'], axis=1)
# Target Series
split_y = customer_features_pd[target_col]

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split train_val and test datasets
X_train, X_val_test, y_train, y_val_test = train_test_split(split_X, split_y, train_size=0.6, random_state=SEED, stratify=split_y)
# Split train and validation datasets
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5, random_state=SEED, stratify=y_val_test)

# Check distribution
dist_arr = []
arr_y = [('training', y_train), ('validation', y_val), ('test', y_test)]

for dataset_name, dist in arr_y:
    c = collections.Counter(dist)
    total = c[0] + c[1]
    dist_arr.append({'dataset':dataset_name, '0': c[0] / total, '1': c[1] / total, 'total': total})

split_ds = pd.DataFrame(dist_arr)
split_ds['%'] = split_ds['total'] / split_ds['total'].sum()
split_ds

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Train classification model
# MAGIC 
# MAGIC Baseline model created by Databricks AutoML:
# MAGIC ```
# MAGIC xgbc_classifier = XGBClassifier(
# MAGIC   colsample_bytree=0.6130041644044038,
# MAGIC   learning_rate=0.020013742401125027,
# MAGIC   max_depth=5,
# MAGIC   min_child_weight=2,
# MAGIC   n_estimators=769,
# MAGIC   n_jobs=100,
# MAGIC   subsample=0.45273300644775333,
# MAGIC   verbosity=0,
# MAGIC   random_state=961644377,
# MAGIC )
# MAGIC ```

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

set_config(display="diagram")

# Scale all feature columns to be centered around zero with unit variance.
standardizer = StandardScaler()

# Model
xgbc_classifier = XGBClassifier(
  colsample_bytree=0.6130041644044038,
  learning_rate=0.020013742401125027,
  max_depth=5,
  min_child_weight=2,
  n_estimators=769,
  n_jobs=100,
  subsample=0.45273300644775333,
  verbosity=0,
  random_state=961644377,
)

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([("standardizer", standardizer)])

# Pipeline Model
model = Pipeline([("standardizer", standardizer), ("classifier", xgbc_classifier)])

model

# COMMAND ----------

mlflow.sklearn.autolog(disable=True)
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost_v4") as mlflow_run:
    model.fit(X_train, y_train, classifier__eval_set=[(X_val_processed, y_val)], classifier__verbose=False)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    #xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")
    # Log metrics for the test set
    #xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")
    
    # Display the logged metrics
    #xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    #xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    #display(pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Train models using the Databricks Feature Store

# COMMAND ----------

from databricks.feature_store import FeatureLookup

key = 'CustomerID'
feature_names = customer_features_df.drop(*['CustomerID', 'Churn']).columns

feature_lookups = [FeatureLookup(table_name=feature_store_db_name_and_table, 
                                 feature_names=feature_names, 
                                 lookup_key=f"{key}")]

with mlflow.start_run(run_name="xgboost_v5") as mlflow_run:

    # Create a training set
    training_set = fs.create_training_set(
        customer_features_df.select(['CustomerID','Churn']),
        feature_lookups = feature_lookups,
        label = 'Churn',
        exclude_columns = ['CustomerID'])

    # Load data from training set
    training_df = training_set.load_df().toPandas()    
    
    X_train = training_df.drop(['Churn'], axis=1)
    y_train = training_df.Churn
    
    model.fit(X_train, y_train, classifier__eval_set=[(X_val_processed, y_val)], classifier__verbose=False)
    
    fs.log_model(model,
                 'train_model',
                  flavor=mlflow.sklearn,
                  training_set=training_set,
                  registered_model_name='churn_prediction')
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    #xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")
    # Log metrics for the test set
    #xgbc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")
    
    # Display the logged metrics
    #xgbc_val_metrics = {k.replace("val_", ""): v for k, v in xgbc_val_metrics.items()}
    #xgbc_test_metrics = {k.replace("test_", ""): v for k, v in xgbc_test_metrics.items()}
    #display(pd.DataFrame([xgbc_val_metrics, xgbc_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Feature importance

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, len(X_train.index)))

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=10)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC 
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC By registering this model in Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC 
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI.

# COMMAND ----------

#run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "xgboost_v4"').iloc[0].run_id
#model_name = "churn_prediction"
#model_version = mlflow.register_model(f"runs:/{run_id}/xgboost_v4", model_name)
