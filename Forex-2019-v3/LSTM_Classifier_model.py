# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:07:20 2019

@author: Chenghai Li
"""

import tensorflow as tf


from LSTM_Base_model import LSTM_Base

class LSTM_Classifier(tf.keras.Model):

  def __init__(self, rate = 0.2, training = True):
    super(LSTM_Classifier, self).__init__(name='LSTM_Base')
    
    predict_model = LSTM_Base(rate = rate, training = training)
    predict_model.load_weights('./checkpoints/LSTM_Base_ckpt')

    self.LSTM1 = predict_model.layers[0]  #tf.keras.layers.LSTM(32, return_sequences=True )
    self.LSTM2 = predict_model.layers[1]  #tf.keras.layers.LSTM(64, return_sequences=False )
    self.FC1 = predict_model.layers[2]  #tf.keras.layers.Dense(16)
    self.enc = predict_model.layers[3]  #AE.layers[0]
    self.FC2 = predict_model.layers[4]  #tf.keras.layers.Dense(8, activation = 'tanh')
    self.FC3 = predict_model.layers[5]  #tf.keras.layers.Dense(1, activation = None)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    
    self.cFC1 = tf.keras.layers.Dense(16)
    self.cdropout1 = tf.keras.layers.Dropout(rate)
    self.cFC2 = tf.keras.layers.Dense(8, activation = 'relu')
    self.cFC3 = tf.keras.layers.Dense(1, activation = 'sigmoid')
    
  def call(self, inputs, training):
      
    
    x = self.LSTM1(inputs)
    x = self.LSTM2(x)
    
    x = self.FC1(x)
    cx = self.cFC1(x)
    
    y = self.enc(inputs)
    
    z = tf.concat([x, y], 1)
    cz = tf.concat([cx, y], 1)
    
    z = self.dropout1(z, training)
    cz = self.cdropout1(cz, training)
    
    z = self.FC2(z)
    z = self.FC3(z)
    
    cz = self.FC2(cz)
    cz = self.FC3(cz)
    
    return z, cz