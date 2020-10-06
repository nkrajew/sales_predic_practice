# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:10:07 2020

@author: nkraj
"""
import numpy as np
import pandas as pd
import datetime
# import random as rd # generating random numbers
# # Viz
# import matplotlib.pyplot as plt # basic plotting
# import seaborn as sns # for prettier plots



# import data
sales = pd.read_csv('sales_train.csv')
item_cat = pd.read_csv('item_categories.csv')
item = pd.read_csv('items.csv')
sub = pd.read_csv('sample_submission.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')

# format date
sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
print(sales.info())

# aggregate data to monthly level
monthly_sales  = sales.groupby(['date_block_num', 'shop_id', 'item_id'])[[
    'date','item_price','item_cnt_day']].agg({'date':['min','max'], 
    'item_price':'mean', 'item_cnt_day':'sum'})
                                              
monthly_sales.head(20)

monthly_sales.to_csv('monthly_sales.csv', index=False)
