# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:30:20 2019

@author: Chenghai Li
"""

import tensorflow as tf
class Encoder(tf.keras.layers.Layer):

  def __init__(self, rate):
    super(Encoder, self).__init__(name = 'Encoder')
    
    self.layer1 = tf.keras.layers.Conv1D( filters=4, kernel_size=3, strides=(1), activation='relu', padding='valid')
    self.layer2 = tf.keras.layers.Conv1D( filters=2, kernel_size=3, strides=(1), activation='relu', padding='valid')
    self.layer3 = tf.keras.layers.Conv1D( filters=1, kernel_size=3, strides=(1), activation='relu', padding='valid')
    self.layer4 = tf.keras.layers.Dense(10, activation = 'tanh')
    
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training):
  
    x = tf.pad(inputs, [[0, 0],[1, 1],[0, 0]], "SYMMETRIC")
    x = self.layer1(x)
    x = tf.keras.layers.AveragePooling1D(2)(x)
    x = tf.pad(x, [[0, 0],[1, 1],[0, 0]], "SYMMETRIC")
    x = self.layer2(x)
    x = tf.keras.layers.AveragePooling1D(2)(x)
    x = tf.pad(x, [[0, 0],[1, 1],[0, 0]], "SYMMETRIC")
    x = self.layer3(x)
    x = tf.keras.layers.Flatten()(x)
    x = self.layer4(x)
    x = self.dropout(x, training)
    
    return x

class Decoder(tf.keras.layers.Layer):

  def __init__(self):
    super(Decoder, self).__init__(name = 'Decoder')
    
    self.layer1 = tf.keras.layers.Dense(16)
    self.layer2 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=(1), activation='relu', padding='valid')
    self.layer3 = tf.keras.layers.Conv1D(filters=2, kernel_size=3, strides=(1), activation='relu', padding='valid')
    self.layer4 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, strides=(1), activation=None, padding='valid')


  def call(self, inputs):
  
    x = self.layer1(inputs)
    x = tf.keras.layers.Reshape(target_shape=(16, 1))(x)
    x = tf.pad(x, [[0, 0],[1, 1],[0, 0]], "SYMMETRIC")
    x = self.layer2(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.pad(x, [[0, 0],[1, 1],[0, 0]], "SYMMETRIC")
    x = self.layer3(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.pad(x, [[0, 0],[1, 1],[0, 0]], "SYMMETRIC")
    x = self.layer4(x)
    
    return x

class AutoEncoder(tf.keras.Model):

  def __init__(self, rate = 0.1, Training = True):
    super(AutoEncoder, self).__init__(name='AutoEncoder')
    
    self.enc = Encoder(rate)
    self.dec = Decoder()
    
  def call(self, inputs, training):
    
    code = self.enc(inputs, training)
    out = self.dec(code)
    
    return out