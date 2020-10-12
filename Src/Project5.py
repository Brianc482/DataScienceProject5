#!/usr/bin/env python
# coding: utf-8

# In[103]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')


demandData = pd.read_csv('C:\\Users\Brian\Desktop\EECS_731\Project5\Data\\Historical Product Demand.csv', index_col=[3], parse_dates=[3])
print(demandData.head())
print("\nThe number of warehouses based on count is:\n")
sns.countplot(x="Warehouse", palette="rocket",data=demandData)
print(demandData["Warehouse"].value_counts(), "\n")
print("Order demands is as follows:\n")
print(demandData["Order_Demand"].value_counts(), "\n")


print("\nThe popularity of products based on count is:\n")
sns.countplot(x="Product_Category", palette="deep", data=demandData)
print(demandData["Product_Category"].value_counts(), "\n")

print(demandData["Product_Code"].value_counts())


demandData["Order_Demand"] = demandData["Order_Demand"].astype(str)
demandData["Order_Demand"] = demandData["Order_Demand"].replace('[(]', '-', regex=True).astype(str)
demandData["Order_Demand"] = demandData["Order_Demand"].replace('[)]', '', regex=True).astype(int)
demandData['Product_Category'] = demandData['Product_Category'].astype(str)
demandData['Product_Category'] = demandData['Product_Category'].replace('[Category_]', '', regex=True).astype(int)

demandData['Product_Code'] = demandData['Product_Code'].astype(str)
demandData['Product_Code'] = demandData['Product_Code'].replace('[Product_]', '', regex=True).astype(int)
print("\nThe order demands are:\n")
print(demandData["Order_Demand"].value_counts())

demandData.groupby([demandData.index.date]).count().Order_Demand.plot(color='red', figsize=(19,5), linewidth=2, markersize=12, title='Numbers Of Orders By Year')

demandData.sort_values(['Date', 'Product_Code', "Order_Demand"], inplace=True)
print(demandData.head(20))


print(demandData.head(20))
demandData = demandData.groupby(['Date','Product_Code', 'Product_Category']).sum().reset_index().set_index('Date')
demandData.sort_values(['Date', 'Product_Code'], inplace=True)

demandData['Weekday'] = demandData.index.map(lambda d : d.weekday())
series = demandData.loc[demandData['Product_Code'] == 349]['Order_Demand']
series = series.resample('D').sum()

training = series.loc[series.index < "2016-01-01"]
testing = series.loc[series.index >= "2015-12-30"]


fig, ax = plt.subplots(figsize=(12,10))
demandData.groupby(['Date','Product_Category']).sum()["Order_Demand"].unstack().plot(ax=ax, color='blue', figsize=(15,5), linewidth=2, markersize=12, animated=True)

trainMean = (training - training.mean())
trainingModel = ARIMA(trainMean, order=(2,0,0))
trainingModelFit = trainingModel.fit()
print(trainingModelFit.summary())

r2Data = training - testing.mean()
predictions = trainingModelFit.predict()
plt.plot(predictions, color="green")
plt.show()


print("The score after applying ARIMA feature engineering based on R^2 regression:")
print(r2_score(r2Data, predictions), "\n")
print("The R2 score subtracted from 100% accuracy gives:")
print(100-r2_score(r2Data, predictions), "\n")


model = AutoReg(training, lags=1)
model_fit = model.fit()
prediction = model_fit.predict(len(testing), len(testing))
print("The results of applying Autoregression: ")
print(prediction, "\n")

model = ARMA(training, order=(2, 1))
model_fit = model.fit(disp=False)
print("The results of applying Autoregressive Moving Average: ")
prediction = model_fit.predict(len(testing), len(testing))
print(prediction, "\n")

model = SARIMAX(training, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))
model_fit = model.fit(disp=False)
print("The results of applying Seasonal Autoregressive Integrated Moving-Average:")
prediction = model_fit.predict(len(testing), len(testing))
print(prediction)