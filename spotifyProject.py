'''
Created on Dec 6, 2017

@author: isaacblinder
'''
import pandas as pd
import numpy as np
import statsmodels as sm
from pandas import datetime
from matplotlib import pyplot as plt
import matplotlib.gridspec as mgrid
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import Series
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_model import ARIMA

#read the csv file
data = pd.read_csv('data.csv')

#rename attributes for easiness of use
data = data.rename(index=str, columns={"Track Name": "trackName"})

#select only specific songs
songNames = ["Safari"]
data = data[data.trackName.isin(songNames)]

dataPiece = data.iloc[:200,:]

#make date the index
index1 = pd.to_datetime(dataPiece['Date'])
index2 = pd.to_datetime(data['Date'])
dataPiece.index = index1
data.index = index2

'''
split_point = len(data) - len(data)*(0.2)
dataset, validation = data[0:split_point], data[split_point:]

print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')
'''

'''
print(data.head())

ts = pd.Series(data["Position"], index = index1)
ts.plot()
plt.show()

x = np.arange(len(data.index))
y = data["Position"]

z = np.polyfit(x, y, 1)
p = np.poly1d(z)


plt.plot(x,y)

plt.plot(x,p(x),"r--")
'''


fig = plt.figure(figsize=(12,6))
grid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])

barax = fig.add_subplot(grid[0])
tsax = fig.add_subplot(grid[1])



dataPiece['Streams'].plot(ax=barax, style='o--')
dataPiece['Position'].plot(ax=tsax, style='o--')
barax.set_ylabel('Streams')
tsax.set_ylabel('Position')


#barax.xaxis.tick_top()
fig.tight_layout()

plt.show()





fig = plt.figure(figsize=(12,6))
grid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])

barax = fig.add_subplot(grid[0])
tsax = fig.add_subplot(grid[1])



data['Streams'].plot(ax=barax, style='o')
data['Position'].plot(ax=tsax, style='o')
barax.set_ylabel('Streams')
tsax.set_ylabel('Position')


#barax.xaxis.tick_top()
fig.tight_layout()

plt.show()



'''
#make a prediction
fig = plt.figure(figsize=(12,6))
dataPiece['Streams'].plot(style='o--')
mod = sm.SARIMAX(dataPiece['Streams'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
'''

# prepare data
X = dataPiece.values[:,0]
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # predict
    yhat = history[-1]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)




# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]





# difference data
months_in_year = 12
stationary = difference(X, months_in_year)
stationary.index = dataPiece.index[months_in_year:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
plt.show()





print('STAGE 3')
# prepare data
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
print (history)
predictions = list()
for i in range(len(test)):
    # difference data
    days_in_week = 7
    diff = difference(history, days_in_week)
    print(len(diff))
    diff.index = index2[:93]
    # predict
    model = ARIMA(diff, order=(1,1,1))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)







