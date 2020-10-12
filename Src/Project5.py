import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import warnings
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
warnings.filterwarnings('ignore')


demandData = pd.read_csv('C:\\Users\Brian\Desktop\EECS_731\Project5\Data\\Historical Product Demand.csv', index_col=[3], parse_dates=[3])
print(demandData.head())
print("\nThe number of warehouses based on count is:\n")
sns.countplot(x="Warehouse", palette="rocket",data=demandData)
print(demandData["Warehouse"].value_counts(), "\n")
print("Order demands is as follows:\n")
print(demandData["Order_Demand"].value_counts(), "\n")

#After reading in the dataset, the first five entries are displayed followed by the total number of warehouses. The number of warehouses is displayed in both a numerical and graphical manner. 
#From these values it is clear to see that "Whse_J" has the highest order demand.

print("\nThe popularity of products based on count is:\n")
sns.countplot(x="Product_Category", palette="deep", data=demandData)
print(demandData["Product_Category"].value_counts(), "\n")

#Whether the graph is vertical or horizontal it is difficult to read the category numbers.
#This is why I have chosed to print the numerical values as well as the graphical representation

print(demandData["Product_Code"].value_counts())

#Since the data is stored with negative values being portrayed by parenthesis, I had to alter the "Order_Demand" column. 
#I was able to portray the values accurately and displayed the results. 

demandData["Order_Demand"] = demandData["Order_Demand"].astype(str)
demandData["Order_Demand"] = demandData["Order_Demand"].replace('[(]', '-', regex=True).astype(str)
demandData["Order_Demand"] = demandData["Order_Demand"].replace('[)]', '', regex=True).astype(int)
demandData['Product_Category'] = demandData['Product_Category'].astype(str)
demandData['Product_Category'] = demandData['Product_Category'].replace('[Category_]', '', regex=True).astype(int)

demandData['Product_Code'] = demandData['Product_Code'].astype(str)
demandData['Product_Code'] = demandData['Product_Code'].replace('[Product_]', '', regex=True).astype(int)
print("The order demands are:\n")
print(demandData["Order_Demand"].value_counts())

#After manipulating the data, I display how many orders the company received per month since 2012.
#There was a large unexplained spike in the data just prior to the year 2014.

demandData.groupby([demandData.index.date]).count().Order_Demand.plot(color='red', figsize=(19,5), linewidth=2, markersize=12, title='Numbers Of Orders By Year')

demandData.sort_values(['Date', 'Product_Code', "Order_Demand"], inplace=True)
print(demandData.head(20))

#At this point, the data has been manipulated and sorted based on date

print(demandData.head(20))
demandData = demandData.groupby(['Date','Product_Code', 'Product_Category']).sum().reset_index().set_index('Date')
demandData.sort_values(['Date', 'Product_Code'], inplace=True)

demandData['Weekday'] = demandData.index.map(lambda d : d.weekday())
series = demandData.loc[demandData['Product_Code'] == 349]['Order_Demand']
series = series.resample('D').sum()

training = series.loc[series.index < "2016-01-01"]
testing = series.loc[series.index >= "2016-01-01"]

fig, ax = plt.subplots(figsize=(12,10))
demandData.groupby(['Date','Product_Category']).sum()["Order_Demand"].unstack().plot(ax=ax, color='blue', figsize=(15,5), linewidth=2, markersize=12, animated=True)

#Setup for the training model and ARIMA feature engineering statistics are displayed

trainMean = (training - training.mean())
trainingModel = ARIMA(trainMean, order=(2,0,0))
trainingModelFit = trainingModel.fit()
print(trainingModelFit.summary())

#Applying an R2 regression score to the acquired data and displaying the information in the form of a graph

r2Data = training - testing.mean()
predictions = trainingModelFit.predict()
plt.plot(predictions, color="green")
plt.show()

print("The score after applying ARIMA feature engineering based on R^2 regression:")
print(r2_score(test2, predictions), "\n")
print("The R2 score subtracted from 100% accuracy gives:")
print(100-r2_score(test2, predictions))

#I am not really sure how to interpret the score that was given. I believe the accuracy is very wrong in either way that it is displayed. I would like to think I was able to attain nearly 100% accuracy, but that is very difficult to believe and in fact it is probably closer to 1%.

