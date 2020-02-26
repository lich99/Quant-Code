# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:08:10 2020

@author: Chenghai Li
"""

import numpy as np
import pandas as pd

data = pd.read_csv('close.csv')
data_index = pd.to_datetime(data['Unnamed: 0'], format='%Y%m%d')
data.index = data_index
data = data.drop(columns=['Unnamed: 0'])
data = data.fillna(method='ffill')

adj = pd.read_csv('adjfactor.csv')
adj_index = pd.to_datetime(adj['Unnamed: 0'], format='%Y%m%d')
adj.index = adj_index
adj = adj.drop(columns=['Unnamed: 0'])
adj = adj.fillna(method='ffill')

data = data * adj

start = data.index[0]
end = data.index[-1]
month_list = pd.date_range(start, end+pd.Timedelta('30 day'), freq='M')
 
stock_list = data.columns
alpha_list = data * 0 / 0

def cal_alpha(alpha_list, N, L, K):
    
    i = -1
    while True:
   
        tar_stock_data = data[:month_list[i]]
        i -= 1
        print(month_list[i])
        if tar_stock_data.shape[0] < L:
            break
        
        temp_data = data[tar_stock_data.index[-K]:tar_stock_data.index[-1]]
        temp_data = temp_data.dropna(axis='columns')
        matrix = temp_data.corr()
        
        for stock in matrix.columns:
            find = matrix.loc[stock].sort_values(ascending=False)[1:N+1]
            
            if find[-1] == "nan":
                continue
            
            ref = temp_data[find.index[0]]   
            for j in range(1, N):
                ref += temp_data[find.index[j]]
            ref /= N
            ps = np.log(temp_data[stock]).fillna(0) - np.log(ref).fillna(0)
            sb = (ps - ps.mean()) / ps.std()
            
            alpha_list[stock][sb.index] = sb.fillna(0)

cal_alpha(alpha_list, 10, 250, 60)
alpha_list.to_csv('out.csv')
    

    