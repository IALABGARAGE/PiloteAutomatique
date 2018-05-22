from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dense, Dropout, Flatten
from keras.optimizers import SGD

from generator import INPUT_SHAPE, batch_generator

import pandas as pd
import numpy as np
import os

batch_size = 32
steps=2000
epochs=10
validation_split = 0.2

def load_data(path):
    data = pd.read_csv(os.path.join(os.getcwd(), path), names=['image', 'angle', 'speed'])

    X = data['image'].values
    Y = data[['angle', 'speed']].values

    return X, Y


def create_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))

    model.add(Conv2D(2, 9, strides=4, activation="relu", padding="same"))
    model.add(Conv2D(4, 5, strides=2, activation="relu", padding="same"))
    model.add(Conv2D(8, 3, strides=2, activation="relu", padding="same"))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='tanh'))

    return model


def train_model(model, X, Y):
    split = int(validation_split*len(X))
    X_train, Y_train = X[split:], Y[split:]
    X_valid, Y_valid = X[:split], Y[:split]


    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')


    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit_generator(batch_generator(X_train, Y_train, batch_size, True),
                        steps,
                        epochs,
                        verbose=1,
                        validation_data=batch_generator(X_valid, Y_valid, batch_size, False),
                        validation_steps=split,
                        callbacks=[checkpoint])


X, Y = load_data('data/data.csv')
model = create_model()

train_model(model, X, Y)