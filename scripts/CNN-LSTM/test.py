
import os
import sys
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import time
import datetime
from utils_laj import *
from data_processing import data_augmentation, analyse_Data
from data_processing import compute_rul_of_one_file

import tensorflow as tf
from keras.layers import LSTM, Dropout, MaxPooling1D, Conv1D
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential

MAXLIFE = 120
SCALE = 1
RESCALE = 1
true_rul = []
test_engine_id = 0
training_engine_id = 0
batch_size = 1024  # Batch size
sequence_length = 100  # Number of steps
n_channels = 24
keep_prob = 0.8

# For LSTM
lstm_size = n_channels * 3  # 3 times the amount of channels
num_layers = 2  # 2  # Number of layers
# Second Dense Layer
ann_hidden = 50     
# Optimizer
learning_rate = 0.001  # 0.0001
epochs = 2  # 5000

# Paths to data
model_directory='./' # directory to save model history after every epoch 
file_path = './CMAPSSData/'
if not ('CMAPSSData' in os.listdir(model_directory)):
    file_path = '../CMAPSSData/'

train_FD001_path = file_path+'train_FD001.txt'
train_FD002_path = file_path+'train_FD002.txt'
train_FD003_path = file_path+'train_FD003.txt'
train_FD004_path = file_path+'train_FD004.txt'


def get_CMAPSSData(save=False, save_training_data=True, files=[1, 2, 3, 4],
                   min_max_norm=False):

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_FD001 = pd.read_table(train_FD001_path, header=None, delim_whitespace=True)
    train_FD002 = pd.read_table(train_FD002_path, header=None, delim_whitespace=True)
    train_FD003 = pd.read_table(train_FD003_path, header=None, delim_whitespace=True)
    train_FD004 = pd.read_table(train_FD004_path, header=None, delim_whitespace=True)
    train_FD001.columns = column_name
    train_FD002.columns = column_name
    train_FD003.columns = column_name
    train_FD004.columns = column_name
    # print("train_FD004.head() \n", train_FD004.head())

    previous_len = 0
    frames = []
    for data_file in ['train_FD00' + str(i) for i in files]:  # load subdataset by subdataset

        #### standard normalization ####
        # print("len(eval(data_file))", len(list(eval(data_file))))
        # print("------------+++++++++++++++")
        # print(eval(data_file).iloc[:, 2:len(list(eval(data_file)))].mean)
        mean = eval(data_file).iloc[:, 2:len(list(eval(data_file)))].mean()
        std = eval(data_file).iloc[:, 2:len(list(eval(data_file)))].std()
        std.replace(0, 1, inplace=True)
        # print("std", std)
        
        ################################

        if min_max_norm:
            scaler = MinMaxScaler()
            eval(data_file).iloc[:, 2:len(list(eval(data_file)))] = scaler.fit_transform(
                eval(data_file).iloc[:, 2:len(list(eval(data_file)))])
        else:
            eval(data_file).iloc[:, 2:len(list(eval(data_file)))] = (eval(data_file).iloc[:, 2:len(
                list(eval(data_file)))] - mean) / std

        eval(data_file)['RUL'] = compute_rul_of_one_file(eval(data_file))
        current_len = len(eval(data_file))
        # print("eval(data_file).index", eval(data_file).index)
        eval(data_file).index = range(previous_len, previous_len + current_len)
        previous_len = previous_len + current_len
        # print(eval(data_file).index)
        frames.append(eval(data_file))

    train = pd.concat(frames)
    # print("++++++++++++")  
    # print("train.head(): \n", train.head(), "\n", "len(train)", len(train))
    
    global training_engine_id
    training_engine_id = train['engine_id']
    train = train.drop('engine_id', 1)
    train = train.drop('cycle', 1)
    # if files[0] == 1 or files[0] == 3:
    #     train = train.drop('setting3', 1)
    #     train = train.drop('s18', 1)
    #     train = train.drop('s19', 1)

    train_values = train.values * SCALE
    # train_values is a numpy representation of the DataFrame.
    # np.save('normalized_train_data.npy', train_values)
    # train.to_csv('normalized_train_data.csv')
    ###########

    return train_values, train


def batch_generator(x_train, y_train, batch_size, sequence_length, online=False, online_shift=1):
    """
    Generator function for creating random batches of training-data for many to many models
    """
    num_x_sensors = x_train.shape[1]
    num_train = x_train.shape[0]
    idx = 0

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_sensors)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)
        # print(idx)
        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            if online == True and (idx >= num_train or (idx + sequence_length) > num_train):
                idx = 0
            elif online == False:
                idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idx:idx + sequence_length]
            y_batch[i] = y_train[idx:idx + sequence_length]
            # print(i,idx)
            if online:
                idx = idx + online_shift  # check if its nee to be idx=idx+1
                # print(idx)
        # print(idx)
        yield (x_batch, y_batch)


training_data, training_pd = get_CMAPSSData()
x_train = training_data[:, :training_data.shape[1] - 1]
print("x_train.shape", x_train.shape)
y_train = training_data[:, training_data.shape[1] - 1]
print("y_train.shape", y_train.shape)

training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
# print(training_generator)
batch_x, batch_y = next(training_generator)
print("batch_x.shape: ", batch_x.shape)
print("$$$$$$$$$$$$$$$$$")
print("batch_y.shape: ", batch_y.shape)


""" -------------- Keras | Ramy -------------- """
model = Sequential()
# augmentation 
# normalization 
model.add(Conv1D(filters=18, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, strides=2, padding='same'))

model.add(Conv1D(filters=36, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, strides=2, padding='same'))

model.add(Conv1D(filters=72, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling1D(2, strides=2, padding='same'))

# shape = conv_last_layer.get_shape().as_list()
# CNN_flat = tf.reshape(conv_last_layer, [-1, shape[1] * shape[2]])
# this will be the input of next Dense layer 

model.add(Flatten())
# Fully connected layer
model.add(Dense(units=sequence_length * n_channels, activation='relu'))
model.add(Dropout(1-keep_prob))

# RESHAPE before LSTM
# lstm_input = tf.reshape(dence_layer_1, [-1, sequence_length, n_channels])
""" TWO TWO TWO TWO LSTM layers - check dropout and sizes and return_sequences """
# model.add(LSTM(units=lstm_size, return_sequences=True))
model.add(LSTM(units=lstm_size, return_sequences=False))

# Reshape before the second dense layer 
# stacked_rnn_output = tf.reshape(rnn_output, [-1, lstm_size])  
""" Dense layer 2 + Dropout """
model.add(Dense(units=ann_hidden, activation='relu'))
model.add(Dropout(1-keep_prob))

# output of the model
model.add(Dense(units=1, activation=None))
model.compile(loss='mse', optimizer='adam',
                  metrics=['mean_absolute_error'])


no_of_batches = int(x_train.shape[0] / batch_size)
print(no_of_batches)
x_train_all = np.zeros((no_of_batches*batch_size, sequence_length, n_channels))
y_train_all = np.zeros((no_of_batches*batch_size, sequence_length))
print("x_train_all.shape", x_train_all.shape)
print("y_train_all.shape", y_train_all.shape)
# batch_x.shape:  (1024, 100, 24)
# batch_y.shape:  (1024, 100)
print("y_train_all.shape", y_train_all.shape)
for btch in range(no_of_batches):
    batch_x, batch_y = next(training_generator)
    x_train_all[btch*batch_size:btch*batch_size+batch_size] = batch_x
    y_train_all[btch*batch_size:btch*batch_size+batch_size] = batch_y

print("x_train_all.shape", x_train_all.shape)
print("y_train_all.shape", y_train_all.shape)



print("Training Model .....")
history = model.fit(x_train_all,
                    y_train_all,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2, shuffle=False)
print(model.summary())

print("Hello here")
sys.exit()



tf.keras.layers.Reshape(target_shape, **kwargs)
https://keras.io/api/layers/reshaping_layers/reshape/
https://keras.io/api/layers/reshaping_layers/flatten/
model.output_shape

# prediction = tf.reshape(output, [-1])
# y_flat = tf.reshape(Y, [-1])
# h = prediction - y_flat
# cost_function = tf.reduce_sum(tf.square(h))
# RMSE = tf.sqrt(tf.reduce_mean(tf.square(h)))
# optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)
# saver = tf.train.Saver()
# training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)

