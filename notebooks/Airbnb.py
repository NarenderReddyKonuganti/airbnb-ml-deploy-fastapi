# Databricks notebook source
# MAGIC %md
# MAGIC # Predict Air B&B dynamic prices
# MAGIC * Download the New York listings.csv.gz from http://insideairbnb.com/get-the-data.html
# MAGIC * Read the uncompressed csv file
# MAGIC * Select a subset of columns for regression
# MAGIC   * you will predict the *price* column
# MAGIC * Cast column values to double or int
# MAGIC   * price will need to be parsed as a double from the currency format (e.g., $100.00)
# MAGIC * Split the data into training and test
# MAGIC * Create a regression model from the training data
# MAGIC * Test the regression model on the test data
# MAGIC * Evaluate the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extract Data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Sources
# MAGIC ##### 1. Airbnb (New York City Data)
# MAGIC ##### 2. Kaggle (Regional Zip codes)

# COMMAND ----------

# MAGIC %fs
# MAGIC cp /FileStore/listings_detail.csv file:/databricks/driver/ --recurse=true 

# COMMAND ----------

# MAGIC %fs
# MAGIC cp /FileStore/nyc_zip.csv file:/databricks/driver/ --recurse=true

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Install required packages

# COMMAND ----------

# MAGIC %pip install shap

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Import the required libraries

# COMMAND ----------

# importing the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree

import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 50)
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create required dataframes csv files

# COMMAND ----------

### Extract Data Part ###
# Import datasets
# Listing 
df = pd.read_csv('listings_detail.csv')
# NYC ZipCode
df.head(5)

# COMMAND ----------

# NYC ZipCode
df_zipcode = pd.read_csv('nyc_zip.csv')
df_zipcode.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Pre-Processing

# COMMAND ----------

# Select Columns
df = df[['id','neighborhood_overview','host_response_rate',
'host_is_superhost','host_total_listings_count','host_has_profile_pic',
'host_identity_verified','zipcode','property_type',
'room_type','accommodates','bathrooms',
'bedrooms','beds','price',
'minimum_nights','maximum_nights','number_of_reviews',
'review_scores_rating','review_scores_accuracy','review_scores_cleanliness',
'review_scores_checkin','review_scores_communication','review_scores_location',
'review_scores_value']]

# Show first 5rows
# df.head()

# Check data types
# df.dtypes

# Count Null data
#df.isnull().sum()

### Cleaning Data ###
# Drop NULL
df_drop_na = df.dropna()

# Remove Strings
df_drop_na['host_response_rate'] = df_drop_na['host_response_rate'].str.strip('%')
df_drop_na['price'] = df_drop_na['price'].str.strip('$')
df_drop_na['price'] = df_drop_na['price'].str.replace(',','')

# Convert boolean to int
arr_mapping = {'t':1, 'f':0}
df_drop_na['host_is_superhost'] = df_drop_na['host_is_superhost'].map(arr_mapping)
df_drop_na['host_has_profile_pic'] = df_drop_na['host_has_profile_pic'].map(arr_mapping)
df_drop_na['host_identity_verified'] = df_drop_na['host_identity_verified'].map(arr_mapping)

# Zipcode - Align 5 numbers
list_zipcode = df_drop_na['zipcode']
new_zipcode = []
for i in list_zipcode:
    if type(i) is str:
        if len(i)<5:
            new_zipcode.append(None)
        else:
            j = i[:5]
            j = int(j)
            new_zipcode.append(j)
    elif type(i) is float:
        j = int(i)
        new_zipcode.append(j)
    else:
        new_zipcode.append(i)

# Switch Zipcode data
df_drop_na['zipcode'] = new_zipcode

#drop rows have NULL data 
df_drop_na = df_drop_na.dropna()

# Change the datatypes
df_drop_na['zipcode'] = df_drop_na['zipcode'].astype(np.int64)
df_drop_na['price'] = df_drop_na['price'].astype(np.float32)

# New Column
df_drop_na['Price_daily'] = df_drop_na['price']/df_drop_na['minimum_nights']

# Merge two datasets
df_merge = pd.merge(df_drop_na, df_zipcode,  left_on='zipcode', right_on='zip' )
df_merge.to_csv('df_merge.csv')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exploratory Data Analysis

# COMMAND ----------

# Count of listing each borough
df_merge.groupby('borough')['id'].count()

# COMMAND ----------

# Average for each Borough 
df_merge.groupby('borough').mean()

# COMMAND ----------

# Count of Listing for each Neighborhood
pd.DataFrame(df_merge.groupby(['borough','neighborhood'])['id'].count())

# COMMAND ----------

# Price Correlation 
pd.DataFrame(df_merge.corr()['Price_daily']).sort_values('Price_daily')

# COMMAND ----------

# To10
pd.DataFrame(df_merge.corr()['Price_daily']).sort_values('Price_daily')[pd.DataFrame(df_merge.corr()['Price_daily']).sort_values('Price_daily', ascending=False)['Price_daily']>0]

# COMMAND ----------

# Review_scores_rating Correlation 
pd.DataFrame(df_merge.corr()['review_scores_rating']).sort_values('review_scores_rating')

# COMMAND ----------

# host_is_superhost Correlation 
pd.DataFrame(df_merge.corr()['host_is_superhost']).sort_values('host_is_superhost')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Correlation Matrix

# COMMAND ----------

# All Corr
df_merge.corr()

# COMMAND ----------

# Frequecy of Review Scores
freq_review_score = df_merge['review_scores_rating'].value_counts()
print(freq_review_score.head())
print('----')
print(freq_review_score.tail())

print('==============')
score_100 = df_merge[df_merge['review_scores_rating']==100.0]
print(score_100['borough'].value_counts())

# COMMAND ----------

scre_under75 = df_merge[df_merge['review_scores_rating']<75.0]
scre_under75.describe()

# COMMAND ----------

sns.regplot(data=df_merge, x='accommodates', y='Price_daily')

# COMMAND ----------

sns.regplot(data=df_merge, x='beds', y='Price_daily')

# COMMAND ----------

sns.regplot(data=df_merge, x='bathrooms', y='Price_daily')

# COMMAND ----------

scre_under75['borough'].value_counts()

# COMMAND ----------

# Highest Daily Price
print(df_merge['Price_daily'].max())

# Highest Price
print(df_merge['price'].max())

# COMMAND ----------

df_merge[df_merge['price']==10000.0]

# COMMAND ----------

df_merge[df_merge['Price_daily']==8000.0]

# COMMAND ----------

df_merge[df_merge['neighborhood']=='Northwest Brooklyn'].describe()

# COMMAND ----------

df_merge[df_merge['neighborhood']=='Northwest Brooklyn'].corr()

# COMMAND ----------

# Price for Highest accommodates
df_merge[df_merge['accommodates']==16]['Price_daily'].describe()

# COMMAND ----------

# Price for Highest bathrooms
df_merge[df_merge['bathrooms']==15.5]['Price_daily'].describe()

# COMMAND ----------

# Price for Highest bathrooms
df_merge[df_merge['beds']==40.0]['Price_daily'].describe()

# COMMAND ----------

# Create Super host datasets
super_host = df_merge[df_merge['host_is_superhost']==1.0]

# COMMAND ----------

# Count of Super Host
super_host['borough'].value_counts()

# COMMAND ----------

# Price Details
super_host['Price_daily'].describe()

# COMMAND ----------

# Score Details
super_host['review_scores_rating'].describe()

# COMMAND ----------

# Count of Super Host with Score100 listings
score_100['host_is_superhost'].value_counts()

# COMMAND ----------

score_100.describe()

# COMMAND ----------

score_100['Price_daily'].describe()

# COMMAND ----------

df=df_merge

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Engineering

# COMMAND ----------

from scipy.special import logit, expit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
#import mlflow
from sklearn.model_selection import RandomizedSearchCV
import os
from IPython.display import Image
from subprocess import call
from sklearn import preprocessing


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Label Encoding

# COMMAND ----------

obj_cols = ['property_type','room_type','borough','neighborhood']
from sklearn.preprocessing import LabelEncoder
label_encoder = preprocessing.LabelEncoder()
for i in range(0, len(obj_cols)) :
    col1 = obj_cols[i]
    df[col1]= label_encoder.fit_transform(df[col1])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Split the dataset
# MAGIC ##### Train - 70%, Test - 30%

# COMMAND ----------

y = pd.DataFrame(df, columns =['Price_daily'])
x = df.drop(columns=['Price_daily','post_office','id','neighborhood_overview','zipcode','population','density','price'])
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30,random_state=3)

# COMMAND ----------

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc
gbc_model = GradientBoostingRegressor()
gbc_model.fit(X_train,Y_train)
Y_pred =gbc_model.predict(X_test)  

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Feature Importance

# COMMAND ----------

feature_importance = gbc_model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
plt.figure(figsize=(8, 18))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.keys()[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### SHAP Analysis

# COMMAND ----------

import shap
explainer = shap.TreeExplainer(gbc_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build ML Models

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Split the dataset
# MAGIC ##### Train - 80%, Test - 20%

# COMMAND ----------

y = pd.DataFrame(df, columns =['Price_daily'])
x = df.drop(columns=['Price_daily','post_office','id','neighborhood_overview','zipcode','population','density','price','host_response_rate'])
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20,random_state=3)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### a. Linear Regression Model

# COMMAND ----------

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# fit the regressor with x and y data
lr.fit(X_train,Y_train)

# COMMAND ----------

# Predict the price
lr_Y_pred = lr.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("Linear Regression Metrics")
print("MAE",mean_absolute_error(Y_test,lr_Y_pred))
print("MSE",mean_squared_error(Y_test,lr_Y_pred))
print("RMSE",np.sqrt(mean_squared_error(Y_test,lr_Y_pred)))
lr_r2 = r2_score(Y_test,lr_Y_pred)
print("R2",lr_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### b. Random Forest Regressor

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 0)
# fit the regressor with x and y data
rfr.fit(X_train,Y_train)

# COMMAND ----------

# Predict the price
rfr_Y_pred = rfr.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("Random Forest Regressor Metrics")
print("MAE",mean_absolute_error(Y_test,rfr_Y_pred))
print("MSE",mean_squared_error(Y_test,rfr_Y_pred))
print("RMSE",np.sqrt(mean_squared_error(Y_test,rfr_Y_pred)))
rfr_r2 = r2_score(Y_test,rfr_Y_pred)
print("R2",rfr_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### c. Gradient Boosting Regressor

# COMMAND ----------

from sklearn import ensemble
gbr =  ensemble.GradientBoostingRegressor()
gbr.fit(X_train,Y_train)

# COMMAND ----------

# Predict the price
gbr_Y_pred = gbr.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("Gradient Regressor Metrics")
print("MAE",mean_absolute_error(Y_test,gbr_Y_pred))
print("MSE",mean_squared_error(Y_test,gbr_Y_pred))
print("RMSE",np.sqrt(mean_squared_error(Y_test,gbr_Y_pred)))
gbr_r2 = r2_score(Y_test,gbr_Y_pred)
print("R2",gbr_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### d. XGBoost Regressor

# COMMAND ----------

y = pd.DataFrame(df, columns =['Price_daily'])
x = df.drop(columns=['Price_daily','post_office','id','neighborhood_overview','zipcode','population','density','price','host_response_rate'])
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20,random_state=3)

# COMMAND ----------

import xgboost as xgb
xgbr = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xgbr.fit(X_train,Y_train)

# COMMAND ----------

# Predict the price
xgbr_Y_pred = model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("XGBoost Regressor Metrics")
print("MAE",mean_absolute_error(Y_test,xgbr_Y_pred))
print("MSE",mean_squared_error(Y_test,xgbr_Y_pred))
print("RMSE",np.sqrt(mean_squared_error(Y_test,xgbr_Y_pred)))
xgbr_r2 = r2_score(Y_test,xgbr_Y_pred)
print("R2",xgbr_r2)

# COMMAND ----------

# evaluate an xgboost regression model on the housing dataset
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(xgbr, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# COMMAND ----------

# Predict the price
xgbr_Y_pred = model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("XGBoost Regressor Metrics")
print("MAE",mean_absolute_error(Y_test,xgbr_Y_pred))
print("MSE",mean_squared_error(Y_test,xgbr_Y_pred))
print("RMSE",np.sqrt(mean_squared_error(Y_test,xgbr_Y_pred)))
xgbr_r2 = r2_score(Y_test,xgbr_Y_pred)
print("R2",xgbr_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC XGBoost Regressor model outperforms other models.
# MAGIC 
# MAGIC We observed that regardless of the price differences and the two different clusters that these listings belong to, their expected demand is relatively close. This means that there is a stable customer ground of each potential location classification: Therefore, it is good to understand where your property stands within the context of the market and customer segments you should aim to target in setting an optimal price. For example, if your listing belongs to a high-end cluster with a high target price, you shouldn’t lower the price too much to attract more customers but instead improve the quality of the listing, investing in amenities to attract more high-end customers. By performing this, you avoid sacrificing profit margins by targeting the wrong customer segment.
