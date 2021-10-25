


import os
import sys
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, MaxPooling1D, Conv1D, Bidirectional
from keras.layers import Activation, Dense, Flatten, Lambda
from keras.models import Sequential

batch_size = 1024

# Read the data
df = pd.read_csv('./household_power_consumption.txt',
                 parse_dates={'dt' : ['Date', 'Time']},
                 sep=";", infer_datetime_format=True,
                 low_memory=False, na_values=['nan','?'], index_col='dt')
# The first five lines of df is shown below
print(df.head())
# we use "dataset_train_actual" for plotting in the end.
dataset_train_actual = df.copy()
# create "dataset_train for further processing
dataset_train = df.copy()


""" Now create training_set which is a 2D numpy array """
# Select features (columns) to be involved intro training and predictions
dataset_train = dataset_train.reset_index()
cols = list(dataset_train)[1:8]
print("cols", cols)

# Extract dates (will be used in visualization)
datelist_train = list(dataset_train['dt'])
datelist_train = [date for date in datelist_train]
training_set = dataset_train[cols].values
print("training_set.shape", training_set.shape)

# Feature Scaling
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])



# Creating a data structure with 72 timestamps and 1 output
X_train = []
y_train = []
n_future = 30 # Number of days we want to predict into the future.
n_past = 72 # Number of past days we want to use to predict future.
for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i,
                   0:dataset_train.shape[1]])
    y_train.append(training_set_scaled[i+n_future-1:i+n_future, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(f'X_train shape == {X_train.shape}')
print(f'y_train shape == {y_train.shape}')

X_train = X_train[:10*batch_size,:,:]
y_train = y_train[:10*batch_size,:]



model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu"
                               ))                           # , input_shape=[None, 7]
# model.add(MaxPooling1D(2, strides=2, padding='same'))

model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Bidirectional(LSTM(32,  return_sequences=False)))
# model.add(Flatten())
model.add(Dense(1))
model.add(keras.layers.Lambda(lambda x: x * 200))

# lr_schedule = keras.callbacks.LearningRateScheduler(
# lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
optimizer=optimizer,
metrics=["mse"])


print(f'X_train shape == {X_train.shape}')
print(f'y_train shape == {y_train.shape}')

history = model.fit(X_train,
                    y_train,
                    epochs=2,
                    batch_size=1024,
                    verbose=2, shuffle=False)
print(model.summary())

