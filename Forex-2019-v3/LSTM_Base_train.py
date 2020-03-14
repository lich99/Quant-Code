# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:11:01 2019

@author: Chenghai Li
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')
y_test = np.load( r'data/y_test.npy')

input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]

BATCH_SIZE = 32
BUFFER_SIZE = x_train_uni.shape[0]

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_univariate = tf.data.Dataset.from_tensor_slices((x_test_uni, y_test_uni))
test_univariate = test_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

from LSTM_Base_model import LSTM_Base

model = LSTM_Base(0.1, True)
model.enc.trainable = False
model.dec.trainable = False

def eval(mode, i):
    global memo
    if mode == 0:
        enc = model.layers[0]
        code = enc(np.expand_dims(x_train_uni[i], axis=0), False)
        hidden = model.layers[0](np.expand_dims(x_train_uni[i], axis=0), False)
        pred = model.layers[1](hidden).numpy()
        plt.clf()
        plt.plot(pred[:,:,-1].flatten())
        plt.plot(x_train_uni[i,:,-1].flatten())
        t = np.arange(3.2,64,6.4)
        plt.scatter(t, code)


optimizer = tf.keras.optimizers.Nadam(0.001)

checkpoint_dir = './checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "LSTM_Base_ckpt")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    monitor = 'val_loss',
    save_best_only = True,
    save_freq = 'epoch',
    save_weights_only = True
    )

EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience = 20,
    baseline = 0.01,
    restore_best_weights = True
    )

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = optimizer,
              metrics=['accuracy']
              )

history = model.fit(train_univariate,
                    epochs = 100,
                    validation_data = test_univariate,
                    #callbacks=[checkpoint_callback, EarlyStopping]
                    )


#model.save_weights('./checkpoints/LSTM_Base_ckpt')
