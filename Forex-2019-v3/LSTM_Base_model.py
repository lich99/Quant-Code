# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:10:24 2019

@author: Chenghai Li
"""

import tensorflow as tf
from AutoEncoder_model import AutoEncoder

AE = AutoEncoder(0.1)
AE.load_weights('./checkpoints/AutoEncoder_ckpt')

class LSTM_Base(tf.keras.Model):

  def __init__(self, rate = 0.2, training = True):
    super(LSTM_Base, self).__init__(name='LSTM_Base')
    
    self.enc = AE.layers[0]
    self.dec = AE.layers[1]
    
    self.LSTM1 = tf.keras.layers.LSTM(32, return_sequences=True )
    self.LSTM2 = tf.keras.layers.LSTM(64, return_sequences=False )
    self.FC1 = tf.keras.layers.Dense(16)
    
    self.FC2 = tf.keras.layers.Dense(8, activation = 'tanh')
    self.FC3 = tf.keras.layers.Dense(3, activation = 'softmax')
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    
  def call(self, inputs, training):
      
    code = self.enc(inputs, training)
    x = self.dec(code)
    x = tf.concat([x, inputs], 2)
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    x = self.FC1(x)
    y = self.enc(inputs, training)
    z = tf.concat([x, y], 1)
    z = self.dropout1(z, training)
    z = self.FC2(z)
    z = self.FC3(z)
    
    return z
