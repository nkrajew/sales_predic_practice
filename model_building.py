# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:38:57 2020

@author: nkraj
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# model libraries
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from data_preprocessing import sales

# establish time series for whole company
ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')

# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,period=12, model='multiplicative')
fig = res.plot()

# additive
res = sm.tsa.seasonal_decompose(ts.values,period=12, model='additive')
fig = res.plot()

# stationarity tests
def test_stationarity(timeseries):
    
    # perform dickey fuller
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value','#Lags used', 'Nubmer of Obs used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(ts)

from pandas import Series as Series

# remove trend
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob

# plot old time series then ts without trend and without seasonality
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()
plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(new_ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()

# test stationarity again after removing seasonality
test_stationarity(new_ts)

# use ARMA model
best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values, order=(i,j)).fit(method='css-mle',trend='nc', solver='nm')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i,j)
                best_model = tmp_mdl
        except: continue

print('AIC: {:6.5f} | order: {}'.format(best_aic, best_order))

# add dates
ts.index=pd.date_range(start = '2013-01-01', end='2015-10-01', freq='MS')
ts=ts.reset_index()
ts.head()

best_mdl.predict()
