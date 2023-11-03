# Databricks notebook source
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre procesamiento de datos con Pandas
# MAGIC Inicialmente se realizara el pre procesamiento de datos utilizandos pandas en el archivo Datamart_Cliente generado del query de integracion, posteriomente para la generacion del modelo se utilizara Spark

# COMMAND ----------

df_berka = spark.read.csv('/FileStore/tables/CasoBerka/Datamart_Cliente.csv', header=True, inferSchema=True)
df_datamart_client = df_berka.toPandas()
df_datamart_client.head()

# COMMAND ----------

df_datamart_client.dtypes 

# COMMAND ----------

# Eliminacion de columnas no necesarias
df_datamart_client = df_datamart_client.drop(['CUENTA_ID'], axis=1)
df_datamart_client = df_datamart_client.drop(['TARJETA_ID'], axis=1)

# Columnas a Object
df_datamart_client["CLIENTE_ID"] = df_datamart_client["CLIENTE_ID"].astype('object')
df_datamart_client["CLIENTE_EDAD"] = df_datamart_client["CLIENTE_EDAD"].astype('object')
df_datamart_client["CLIENTE_GENERO"] = df_datamart_client["CLIENTE_GENERO"].astype('object')
df_datamart_client["CLIENTE_RATIO_DESEMPLEO"] = df_datamart_client["CLIENTE_RATIO_DESEMPLEO"].astype('object')
df_datamart_client["CLIENTE_REGION"] = df_datamart_client["CLIENTE_REGION"].astype('object')
df_datamart_client["CUENTA_ANTIGUEDAD"] = df_datamart_client["CUENTA_ANTIGUEDAD"].astype('object')
df_datamart_client["DEUDA_CATEGORIA"] = df_datamart_client["DEUDA_CATEGORIA"].astype('object')
df_datamart_client["TARJETA_CREDITO"] = df_datamart_client["TARJETA_CREDITO"].astype('object')

# COMMAND ----------

df_datamart_client.describe().T

# COMMAND ----------

df_datamart_client.dtypes

# COMMAND ----------

len(df_datamart_client)

# COMMAND ----------

ax = df_datamart_client['CLIENTE_SALARIO_PROMEDIO'].hist(bins=50, color = 'lightblue', ec = 'white', figsize=(15, 5))
plt.title('Histograma de CLIENTE_SALARIO_PROMEDIO')
plt.show()

# COMMAND ----------

plt.figure(figsize = (15,3))
plt.subplot(121)
plt.title('Cantidad de Clientes por Categoria de Edad')
df_datamart_client.groupby('CLIENTE_EDAD')['CLIENTE_ID'].nunique().plot(kind='barh', color = 'lightcoral', ec = 'white')
plt.subplot(122)
plt.title('Cantidad de Clientes por Genero')
df_datamart_client.groupby('CLIENTE_GENERO')['CLIENTE_ID'].nunique().plot(kind='barh', color = 'teal', ec = 'white')
plt.show()

# COMMAND ----------

df_datamart_client[['DEPOSITO_MONTOAVG', 'RETIRO_MONTOAVG']].plot(kind='hist', bins=25, alpha=0.4, figsize=(15,4), ec = 'white')
plt.title('Histograma Monto AVG Deposito vs Monto AVG Retiro')

# COMMAND ----------

df_datamart_client[['DEPOSITO', 'RETIRO']].plot(kind='hist', bins=25, alpha=0.4, figsize=(15,4), ec = 'white')
plt.title('Histograma Cantidad de Depositos')

# COMMAND ----------

df_datamart_client[['DEPOSITO', 'RETIRO']].plot(kind='hist', bins=25, alpha=0.4, figsize=(15,4), ec = 'white')
plt.title('Histograma Cantidad de Depositos')

# COMMAND ----------

plt.figure(figsize = (15,3))
plt.subplot(121)
df_datamart_client.groupby('DEUDA_CATEGORIA')['CLIENTE_ID'].nunique().plot(kind='barh', color = 'plum', ec = 'white')
plt.subplot(122)
df_datamart_client.groupby('CUENTA_ANTIGUEDAD')['CLIENTE_ID'].nunique().plot(kind='barh', color = 'skyblue', ec = 'white')
plt.show()

# COMMAND ----------

df_datamart_client['SALDO_AVG'].hist(bins=50, color = 'steelblue', ec = 'white', figsize=(15, 5))
plt.title('Histograma de Salario Promedio de Clientes')
plt.show()

# COMMAND ----------

df_datamart_client['TOTAL_TRANSACCIONES'].plot(kind = 'kde', color = 'lightseagreen', figsize=(15, 5))
plt.title('Distribucion del Total de Transacciones')
plt.show()

# COMMAND ----------

corr_df = df_datamart_client.corr(method='pearson')

plt.figure(figsize=(13, 10))
sns.heatmap(corr_df, annot=True)
plt.show()

# COMMAND ----------

# Analisis Bivariado
df_datamart_client.groupby(['TARJETA_CREDITO']).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean
                                       }).sort_values(['TARJETA_CREDITO'], ascending = True ) 

# COMMAND ----------

df_datamart_client.groupby(['CLIENTE_EDAD','TARJETA_CREDITO'], dropna=False).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean})

# COMMAND ----------

df_datamart_client.groupby(['CLIENTE_GENERO','TARJETA_CREDITO'], dropna=False).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean})

# COMMAND ----------

df_datamart_client.groupby(['DEUDA_CATEGORIA','TARJETA_CREDITO'], dropna=False).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean})

# COMMAND ----------

df_datamart_client.groupby(['CLIENTE_REGION','TARJETA_CREDITO'], dropna=False).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean})

# COMMAND ----------

df_datamart_client.groupby(['CLIENTE_RATIO_DESEMPLEO','TARJETA_CREDITO'], dropna=False).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean})

# COMMAND ----------

df_datamart_client.groupby(['CUENTA_ANTIGUEDAD','TARJETA_CREDITO'], dropna=False).agg({'CLIENTE_SALARIO_PROMEDIO':np.mean, 'TOTAL_TRANSACCIONES':np.mean, 'DEPOSITO':np.mean, 'RETIRO':np.mean, \
                                                      'SALDO_AVG':np.mean, 'DEPOSITO_MONTOAVG':np.mean, 'RETIRO_MONTOAVG':np.mean})

# COMMAND ----------

df_datamart_client.columns

# COMMAND ----------

#Tratamiento de Datos Nulos
df_datamart_client.isnull().sum()

# COMMAND ----------

# Al haberse realizado un pre procesamiento de los datos en SQL, en esta ocasion no se encontraron datos nulos a ser procesados
sns.heatmap(df_datamart_client.isnull(), cbar=False)

# COMMAND ----------

df_datamart_client.shape

# COMMAND ----------

# Tratamiento de Datos Atipicos con Z-Score
features_Zscore = ['CLIENTE_SALARIO_PROMEDIO', 'TOTAL_TRANSACCIONES', 'DEPOSITO'
                   , 'RETIRO', 'SALDO_AVG', 'DEPOSITO_MONTOAVG', 'RETIRO_MONTOAVG']

z = np.abs(stats.zscore(df_datamart_client[['CLIENTE_SALARIO_PROMEDIO', 'TOTAL_TRANSACCIONES', 'DEPOSITO'
                                            , 'RETIRO', 'SALDO_AVG', 'DEPOSITO_MONTOAVG', 'RETIRO_MONTOAVG']]))

(z<3).all(axis=1)    
df_datamart_client = df_datamart_client[(z<3).all(axis=1)]

# Reduccion de 4500 clientes a 4407
df_datamart_client.shape

# COMMAND ----------

df_datamart_client.head()

# COMMAND ----------

# Variables a ser reescaladas con MinMax Encoder
features_mimmax = ['CLIENTE_SALARIO_PROMEDIO', 'TOTAL_TRANSACCIONES', 'DEPOSITO', 'RETIRO', 'SALDO_AVG', 'DEPOSITO_MONTOAVG', 'RETIRO_MONTOAVG']

objeto_scaler = MinMaxScaler()
df_datamart_client[features_mimmax] = objeto_scaler.fit_transform(df_datamart_client[features_mimmax])
df_datamart_client.head()
