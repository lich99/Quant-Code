# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:20:09 2019

@author: Chenghai Li
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')
y_test = np.load( r'data/y_test.npy')
y_train_pip = np.load( r'data/y_train_pip.npy')
y_test_pip = np.load( r'data/y_test_pip.npy')


input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]

from LSTM_Base_model import LSTM_Base

model = LSTM_Base(0.1, training = False)
model.load_weights('./checkpoints/LSTM_Base_ckpt')


def binary_acc(mode, threshold):
    
    right = 0
    all_num = 0
    if mode == 0:
        pred = model.predict( x_train_uni )
        label = y_train_uni
    if mode == 1:
        pred = model.predict( x_test_uni )
        label = y_test_uni
    for i in range ( len( pred) ):
        if pred[i][0] > threshold: 
            if label[i][0] > 0:
                right += 1
            all_num += 1
        if pred[i][0] < threshold: 
            if label[i][0] < 0:
                right += 1
            all_num += 1
    return right / all_num

def profit(mode, threshold):
    
    plist = []
    profit = 0
    if mode == 0:
        pred = model.predict( x_train_uni )
        label = y_train_pip
    if mode == 1:
        pred = model.predict( x_test_uni )
        label = y_test_pip
    for i in range ( len( pred ) ):
        
        check = 0
        if pred[i] > threshold:
            profit += label[i][0]
            check = 1
        if pred[i] < -threshold:
            profit-= label[i][0]
            check = 1
        if check == 1:
            profit -= 0.00005
            
        plist.append(profit)
    plt.plot(plist)
    return profit

def profit_with_weight(mode, threshold):
    
    plist = []
    profit = 0
    if mode == 0:
        pred = model.predict( x_train_uni )
        label = y_train_pip
    if mode == 1:
        pred = model.predict( x_test_uni )
        label = y_test_pip
    for i in range ( len( pred ) ):
        if abs(pred[i][0]) < threshold:
            continue

        profit += label[i][0] * pred[i][0] * 10000
        profit -= abs(pred[i][0]) * 0.00005 * 10000
        plist.append(profit)
        
    plt.plot(plist)
    return profit

#plt.hist(y_train_uni.flatten(), bins=200, color='steelblue', normed=True )
profit(1, 0.025)

def cross_test():
    test_predictions = model.predict(x_test_uni).flatten()

    plt.axes(aspect='equal')
    plt.scatter(y_test_pip, test_predictions)
    
    lims = [-0.05, 0.05]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)

def show_error():
    label = np.copy(y_test_pip)
    test_predictions = model.predict(x_test_uni)
    test_predictions /= test_predictions.std()
    test_predictions = test_predictions.flatten()
    label /= label.std()
    label = label.flatten()
    error = test_predictions - label
    plt.hist(error, bins = 100)
   
from sklearn.metrics import roc_curve

def show_roc():
    
    label = np.copy(y_test_pip)
    test_predictions = model.predict(x_test_uni)
    test_predictions /= test_predictions.std()
    test_predictions = test_predictions.flatten()
    label /= label.std()
    label = label.flatten()
    
    test_predictions[test_predictions > 0] = 1
    test_predictions[test_predictions <= 0] = 0
    label[label > 0] = 1
    label[label <= 0] = 0
    
    fpr, tpr, _ = roc_curve(label, test_predictions)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1])
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    #plt.axes(aspect='equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()