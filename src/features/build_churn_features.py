# Databricks notebook source
# MAGIC %run ../config_env

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. About dataset
# MAGIC 
# MAGIC The data set belongs to a leading online E-Commerce company. An online retail (E-commerce) company wants to know the customers who are going to churn, so accordingly they can approach customer to offer some promos.
# MAGIC 
# MAGIC - **CustomerID**: Unique customer ID
# MAGIC - **Tenure**: Tenure of customer in organization
# MAGIC - **PreferredLoginDevice**: Preferred login device of customer
# MAGIC - **CityTier**: City tier
# MAGIC - **WarehouseToHome**: Distance in between warehouse to home of customer
# MAGIC - **PreferredPaymentMode**: Preferred payment method of customer
# MAGIC - **Gender**: Gender of customer
# MAGIC - **HourSpendOnApp**: Number of hours spend on mobile application or website
# MAGIC - **NumberOfDeviceRegistered**: Total number of deceives is registered on particular customer
# MAGIC - **PreferedOrderCat**: Preferred order category of customer in last month
# MAGIC - **SatisfactionScore**: Satisfactory score of customer on service
# MAGIC - **MaritalStatus**: Marital status of customer
# MAGIC - **NumberOfAddress**: Total number of addresses added on particular customer
# MAGIC - **Complain**: Any complaint has been raised in last month
# MAGIC - **OrderAmountHikeFromlastYear**: Percentage increases in order from last year
# MAGIC - **CouponUsed**: Total number of coupon has been used in last month
# MAGIC - **OrderCount**: Total number of orders has been places in last month
# MAGIC - **DaySinceLastOrder**: Day Since last order by customer
# MAGIC - **CashbackAmount**: Average cashback in last month
# MAGIC - **Churn**: Churn Flag

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Libraries

# COMMAND ----------

#!pip3 install openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql.types import _parse_datatype_string
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Loading data

# COMMAND ----------

df_raw = pd.read_excel(path_raw_data, sheet_name='E Comm')

# Copy of data
df = df_raw.copy()
df.shape, df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Exploratory Data Analysis
# MAGIC 
# MAGIC The dataset contains 5.630 rows and 20 columns. We analysed all the dataset and didn't find duplicated rows and incorrect data types.
# MAGIC Below, we list some steps considered during exploratory data analysis:
# MAGIC - Duplicated rows analysis
# MAGIC - Incorrect data types analysis
# MAGIC - Need to normalization of categorical variables
# MAGIC - Outliers analysis

# COMMAND ----------

display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Processing: Null values**
# MAGIC 
# MAGIC Some variables have null values, we analysed and decided using the median values to fill nulls

# COMMAND ----------

# Null Values
null_cols = [col for col in df.columns if df[col].isnull().sum()]

null_p = df[null_cols].isnull().sum().reset_index()
null_p.columns = ['Columns','#']
null_p['%'] = round((null_p['#'] / df.shape[0]) * 100, 3)

display(null_p)
display(df[null_cols].describe().transpose().reset_index())

# COMMAND ----------

# Fill null columns with median values
[df[col].fillna(df[col].median(), inplace=True) for col in df.columns if df[col].isnull().sum()]

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Processing: Outliers**
# MAGIC 
# MAGIC In a preliminary analysis we decide to use the IQR method to detect outliers because it is very simple to understand and use. After calc IQR distance and detect outliers, we'll fill the values with the range IQR.
# MAGIC 
# MAGIC Method: 1,5 * IQR

# COMMAND ----------

def calc_outlier_range(col):
    times_IQR = 1.5                          # Define how many times IQR will used to calc Lower and Upper Range
    sorted(col)                              # Sorted series object
    Q1,Q3 = np.percentile(col,[25,75])       # Calc Q1 and Q3
    IQR = Q3-Q1                              # Calc the distance between Q3 and Q1
    lr= Q1-(times_IQR * IQR)                 # Calc Lower Range
    ur= Q3+(times_IQR * IQR)                 # Calc Upper Range
    
    return lr, ur

# COMMAND ----------

plt.figure(figsize=(50,10))
sns.boxplot(data=df.drop(columns=['CustomerID','Churn','CityTier'])).set_title('Outlier Analysis')

# COMMAND ----------

# Drop CustomerID, Churn and CityTier columns and select all that dtype is not object
columns_to_detect_outliers = [col for col in df.drop(columns=['CustomerID','Churn','CityTier'], axis=1).columns if df[col].dtype != 'object']

for column in columns_to_detect_outliers:
    # Get range IQR
    lr, ur = calc_outlier_range(df[column])
    
    # Fill outlier values with Upper Range or Lower Range
    df[column]=np.where(df[column]>ur,ur,df[column])
    df[column]=np.where(df[column]<lr,lr,df[column])

# COMMAND ----------

data = df.drop(columns=['CustomerID','Churn','CityTier'])
plt.figure(figsize=(50,10))
sns.boxplot(data=data).set_title('Outlier Analysis - After fix outlier values')

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Processing: Casting types**
# MAGIC 
# MAGIC Casting to object: CityTier and Churn

# COMMAND ----------

df['Churn'] = df['Churn'].astype('object')
df['CityTier'] = df['CityTier'].astype('object')

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Processing: OneHotEncoding**

# COMMAND ----------

# Select cols to apply OneHotEncoder
cols_to_encoder = [col for col in df.drop(columns=['Churn']).columns if df[col].dtype == 'object']

# Create and fit encoder
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(df[cols_to_encoder])

# Transform features
features = one_hot_encoder.transform(df[cols_to_encoder]).toarray()
features_col_name = one_hot_encoder.get_feature_names(cols_to_encoder)

# Dataframe with encode features
df_features_encoder = pd.DataFrame(data=features, columns=features_col_name)
df_features_encoder['CustomerID'] = df['CustomerID']

# Final dataframe
df_features = df.copy()
df_features.drop(columns=cols_to_encoder, inplace=True, axis=1)
df_features = df_features.merge(df_features_encoder, on=['CustomerID'], how='inner')

df_features.shape

# COMMAND ----------

df_features.columns

# COMMAND ----------

display(df_features.head())

# COMMAND ----------

# Rename column names to remove special chars
cols = [col.replace(' ','_').replace('&','and') for col in df_features.columns]
df_features.columns = cols

# COMMAND ----------

# MAGIC %md
# MAGIC **Testing schema before write on the feature tables**

# COMMAND ----------

feature_schema = """
    CustomerID long,
    Churn long,
    Tenure double,
    WarehouseToHome double,
    HourSpendOnApp double,
    NumberOfDeviceRegistered double,
    SatisfactionScore double,
    NumberOfAddress double,
    Complain double,
    OrderAmountHikeFromlastYear double,
    CouponUsed double,
    OrderCount double,
    DaySinceLastOrder double,
    CashbackAmount double,
    PreferredLoginDevice_Computer double,
    PreferredLoginDevice_Mobile_Phone double,
    PreferredLoginDevice_Phone double,
    CityTier_1 double,
    CityTier_2 double,
    CityTier_3 double,
    PreferredPaymentMode_CC double,
    PreferredPaymentMode_COD double,
    PreferredPaymentMode_Cash_on_Delivery double,
    PreferredPaymentMode_Credit_Card double,
    PreferredPaymentMode_Debit_Card double,
    PreferredPaymentMode_E_wallet double,
    PreferredPaymentMode_UPI double,
    Gender_Female double,
    Gender_Male double,
    PreferedOrderCat_Fashion double,
    PreferedOrderCat_Grocery double,
    PreferedOrderCat_Laptop_and_Accessory double,
    PreferedOrderCat_Mobile double,
    PreferedOrderCat_Mobile_Phone double,
    PreferedOrderCat_Others double,
    MaritalStatus_Divorced double,
    MaritalStatus_Married double,
    MaritalStatus_Single double
"""

assert spark.createDataFrame(df_features).schema == _parse_datatype_string(feature_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Feature Store
# MAGIC 
# MAGIC In order to do this, we'll want to provide the following:
# MAGIC 1. The **name** of the database and table where we want to store the feature store
# MAGIC 2. The **keys** for the table
# MAGIC 3. The **schema** of the table
# MAGIC 4. A **description** of the contents of the feature store
# MAGIC 
# MAGIC Create the feature table

# COMMAND ----------

feature_store_db_name

# COMMAND ----------

sqlContext.sql(f"CREATE DATABASE IF NOT EXISTS {feature_store_db_name};")

# COMMAND ----------

df_spark_features = spark.createDataFrame(df_features)

# COMMAND ----------

feature_store_db_name_and_table

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

try:
    fs.get_table(feature_store_db_name_and_table)
    print(f'### There is a feature table named {feature_store_db_name_and_table}')
except:
    feature_table = fs.create_table(
        name=feature_store_db_name_and_table,
        primary_keys=["CustomerID"],
        df=df_spark_features,
        description="This table contains one-hot and numeric features to predict the churn of a customer")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Write the records to the feature table

# COMMAND ----------

fs.write_table(df=df_spark_features, name=feature_store_db_name_and_table, mode='merge') #mode='merge/overwrite'
