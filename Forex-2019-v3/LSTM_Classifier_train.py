# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:31:07 2019

@author: Chenghai Li
"""
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')
y_test = np.load( r'data/y_test.npy')
y_test_pip = np.load( r'data/y_test_pip.npy')

input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]


BATCH_SIZE = 64
BUFFER_SIZE = x_train_uni.shape[0]

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_univariate = tf.data.Dataset.from_tensor_slices((x_test_uni, y_test_uni))
test_univariate = test_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

from LSTM_Classifier_model import LSTM_Classifier

model = LSTM_Classifier(training = True)

optimizer = tf.keras.optimizers.Adam(0.0001)

def loss_function(pred, predc, tar):
    
   
  return -tf.reduce_mean(pred * predc * tar) + tf.reduce_mean(tf.abs(predc * pred)) * 0.005

train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')

@tf.function()
def train_step(inp, tar):

  with tf.GradientTape() as tape:
      
    predictions, predictionsc = model(inp, True)
    loss = loss_function(predictions, predictionsc , tar)

  gradients = tape.gradient(loss, model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  
@tf.function()
def val_step(inp, tar):
    
  predictions, predictionsc = model(inp, False)
  loss = loss_function(predictions, predictionsc , tar)
  val_loss(loss)

EPOCHS = 100

for epoch in range(EPOCHS):
  start = time.time()
  
  train_loss.reset_states()
  val_loss.reset_states()
  
  for (batch, (inp, tar)) in enumerate(train_univariate):  
    train_step(inp, tar)

  for (batch, (inp, tar)) in enumerate(train_univariate):  
    val_step(inp, tar)
  

  print ('Epoch {} Loss {:.10f} Val_Loss {:.10f}'.format(epoch + 1,train_loss.result(), val_loss.result()))
  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))