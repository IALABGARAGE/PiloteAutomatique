# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:12:48 2018

@author: matth --- 15600
"""
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

from keras.optimizers import SGD
import matplotlib.pyplot as plt
from generator import INPUT_SHAPE, batch_generator

import numpy as np
import pandas as pd
from numpy import newaxis,array

input = pd.read_csv("imgdataset.csv").values
output = pd.read_csv("valeursvt.csv").values


print ("Importation du dataset")

x_train = input [60:600,:]
x_test = input [500 : 750]
y_train = output [60:600,:]
y_test = output [500 : 750]

img = x_train[100:101,:]
img = img.reshape(120,185)
plt.imshow(img)
print (img.shape)
img2 = y_train[100:101,:]
print (img2.shape)
print (x_train)

x_train = x_train[:,0:].reshape(x_train.shape[0],1,185,120).astype( 'float32' )
x_test = x_test[:, 0:].reshape(x_test.shape[0],1,185,120).astype( 'float32' )


print(" X_train : {}".format( x_train.shape))
print(" Y_train : {}".format(y_train.shape))
print(" X_test : {}".format(x_test.shape))
print(" Y_test : {}".format(y_test.shape))


batch_size = 32
steps=2000
epochs=10
validation_split = 0.2
inputshape= (1,120,185)


model = Sequential()
K.set_image_dim_ordering('th')
model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(1, 185, 120),activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(15, 3, 3, activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu' ))
model.add(Dense(50, activation= 'relu' ))
model.add(Dense(2, activation= 'softmax' ))
  # Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

model.fit(x_train, y_train,
          epochs=20,
          batch_size= 160)

score = model.evaluate(x_test, y_test, batch_size=128)

# 10. Evaluate model on test data
#prediction = model.predict(x_train[60:61,:])
#print(prediction)

