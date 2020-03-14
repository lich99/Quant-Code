# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:27:46 2019

@author: Chenghai Li
"""

import csv
from args import args
import numpy as np
from talib import RSI
from matplotlib import pyplot as plt

arg = args()

step = arg.step
input_length = arg.input_length
predict_length = 5
split_ratio = arg.split_ratio

file = open(r'data/EURUSD.csv')
file_csv = csv.reader(file)
data = []

for row in file_csv:
    row[2] = float(row[2])
    row[3] = float(row[3])
    row[4] = float(row[4])
    row[5] = float(row[5])
    data.append(row[2 : 6])
    
data = np.array(data)
index = int(len(data) * split_ratio)

train = np.array(data[0 : index], dtype = np.float32)
test = np.array(data[index : ], dtype = np.float32)
print('All data:', len(data), '  Train:', len(train), '  Test:', len(test))

x_train = []
y_train = []

for i in range (0, len(train) - input_length - predict_length, step):
    x_train.append(train[i : i + input_length])
    y_train.append(train[i + input_length : i + input_length + predict_length, -1])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train_uni = np.zeros_like(x_train)
y_train_uni = np.zeros_like(y_train)
y_train_check = np.zeros_like(y_train)
y_train_pip = np.zeros((x_train.shape[0], 1))

for i in range(len(x_train)):
    
    s = x_train[i][0][0]
    x_train_uni[i] = x_train[i] / s * 100
    y_train_uni[i] = y_train[i] / s * 100
    d = x_train_uni[i].mean()
    x_train_uni[i] -= d
    y_train_uni[i] -= (d + x_train_uni[i][-1][-1])
    y_train_check[i] = y_train_uni[i]
    y_train_pip[i] = y_train[i].mean() - x_train[i][-1][-1]
    
y_train_mean = np.mean(y_train_uni, 1)
y_train_max = np.percentile(y_train_mean.flatten(), 80)
y_train_min = np.percentile(y_train_mean.flatten(), 20)

for i in range(len(y_train_mean)):
    
    if y_train_mean[i] > y_train_max:
        y_train_mean[i] = 2
    elif y_train_mean[i] < y_train_min:
        y_train_mean[i] = 0
    else:
        y_train_mean[i] = 1
      
#plt.hist(y_train_uni.flatten(), bins=200, color='steelblue', normed=True )
            
x_test = []
y_test = []

for i in range (0, len(test) - input_length - 10, step):
    x_test.append(test[i : i + input_length])
    y_test.append(test[i + input_length : i + input_length + 10, -1])

x_test = np.array(x_test, dtype = np.float32)
y_test = np.array(y_test, dtype = np.float32)

x_test_uni = np.zeros_like(x_test)
y_test_uni = np.zeros_like(y_test)
y_test_pip = np.zeros((x_test.shape[0], 1))

for i in range(len(x_test)):
    
    s = x_test[i][0][0]
    x_test_uni[i] = x_test[i] / s * 100
    y_test_uni[i] = y_test[i] / s * 100
    d = x_test_uni[i].mean()
    x_test_uni[i] -= d
    y_test_uni[i] -= (d + x_test_uni[i][-1][-1])
    y_test_pip[i] = y_test[i].mean() - x_test[i][-1][-1]
    

y_test_mean = np.mean(y_test_uni, 1)

y_test_max = np.percentile(y_test_mean.flatten(), 80)
y_test_min = np.percentile(y_test_mean.flatten(), 20)

for i in range(len(y_test_mean)):
    
    if y_test_mean[i] > y_test_max:
        y_test_mean[i] = 2
    elif y_test_mean[i] < y_test_min:
        y_test_mean[i] = 0
    else:
        y_test_mean[i] = 1

y_train_mean = np.expand_dims(y_train_mean, 1)
y_test_mean = np.expand_dims(y_test_mean, 1)

np.save( r'data/x_train_uni.npy', x_train_uni)
np.save( r'data/y_train_uni.npy', y_train_mean)
np.save( r'data/x_test_uni.npy', x_test_uni)
np.save( r'data/y_test_uni.npy', y_test_mean)
np.save( r'data/y_test.npy', y_test)
np.save( r'data/y_test_pip.npy', y_test_pip)
np.save( r'data/y_train_pip.npy', y_train_pip)

print('Processed data:', len(x_train_uni) + len(x_test_uni), '  Train:', len(x_train_uni), '  Test:', len(x_test_uni))
print('Saved!')