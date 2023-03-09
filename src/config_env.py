# Databricks notebook source
path_raw_data = '/dbfs/FileStore/E_Commerce_Dataset.xlsx'

# COMMAND ----------

# Database Feature Store and Feature Table
feature_store_db_name = 'fs_ecommerce_<MEU-EMAIL>'
feature_store_db_name_and_table = 'fs_ecommerce_<MEU-EMAIL>.churn'

# COMMAND ----------

# Criando camada diamante
sqlContext.sql(f"CREATE DATABASE IF NOT EXISTS diamond;")
