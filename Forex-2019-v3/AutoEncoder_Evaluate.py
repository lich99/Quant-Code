# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 23:32:18 2019

@author: Chenghai Li
"""

import numpy as np
from matplotlib import pyplot as plt

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')
y_test = np.load( r'data/y_test.npy')
y_test_pip = np.load( r'data/y_test_pip.npy')

input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]

from AutoEncoder_model import AutoEncoder

model = AutoEncoder()
model.load_weights('./checkpoints/AutoEncoder_ckpt')

memo = 0
def eval(mode, i):
    global memo
    if mode == 0:
        enc = model.layers[0]
        code = enc(np.expand_dims(x_train_uni[i], axis=0), False)
        pred = model.predict(np.expand_dims(x_train_uni[i], axis=0))
        plt.clf()
        plt.plot(pred[:,:,-1].flatten())
        plt.plot(x_train_uni[i,:,-1].flatten())
        t = np.arange(3.2,64,6.4)
        plt.scatter(t, code)
        memo = code
    if mode == 1:
        pred = model.predict(np.expand_dims(x_test_uni[i], axis=0))
        plt.clf()
        plt.plot(pred[:,:,-1].flatten())
        plt.plot(pred[:,:,0].flatten())
        plt.plot(pred[:,:,1].flatten())
        plt.plot(pred[:,:,2].flatten())
        plt.plot(x_test_uni[i,:,-1].flatten())
    
eval(0, 566)
