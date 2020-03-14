# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:42:34 2019

@author: Chenghai Li
"""

import numpy as np
import tensorflow as tf
import os

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')

input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]

BATCH_SIZE = 64
BUFFER_SIZE = x_train_uni.shape[0]

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, x_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_univariate = tf.data.Dataset.from_tensor_slices((x_test_uni, x_test_uni))
test_univariate = test_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


from AutoEncoder_model import AutoEncoder

AE = AutoEncoder(0.1, True)

optimizer = tf.keras.optimizers.Adam(0.001)

checkpoint_dir = './checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "AutoEncoder_ckpt")

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

AE.compile(loss = 'mse',
              optimizer = optimizer,
              )

history = AE.fit(train_univariate,
                    epochs = 100,
                    validation_data = test_univariate,
                    callbacks=[checkpoint_callback, EarlyStopping]
                    )

model.save_weights('./checkpoints/ckpt')



