
import datetime
import sys
import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from random import randint
import requests
import zipfile
from io import StringIO
import os
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from load_data import load_data

pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = datetime.datetime.now()
RUL_cap = 130


model_directory='./' # directory to save model history after every epoch 
file_path = './CMAPSSData/'
if not ('CMAPSSData' in os.listdir(model_directory)):
    file_path = './scripts/CMAPSSData/'

train_file = file_path+'train_FD001.txt'
test_file = file_path+'test_FD001.txt'
rul_file = file_path+'RUL_FD001.txt'

column_names = ["2D features 29*17", "RUL"]


# convert series to supervised learning
def series_to_supervised(data, window_size, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    # print("n_vars", n_vars)  # 18
    df = data
    df1 = df.drop(['RUL'], axis=1)
    cols = list()
    for k in range(df1.shape[0]-window_size):
        df1_to_pic = df1.iloc[k:k+window_size][:]
        cols.append([df1_to_pic.values, df.iloc[k+window_size-1]['RUL']])
    agg = pd.DataFrame(cols, columns = column_names)   
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_train_data():
    training_data = load_data(train_file)
    num_engine = max(training_data['engine_no'])
    test_data = load_data(test_file)

    max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
    window_size = max_window_size - 2

    # Follow variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19 for FD001
    training_data = training_data.drop(
        ['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1)  # FD001
    
    # df is the training data
    df_train = pd.DataFrame()
    for i in range(num_engine):
        # print("i+1: ", i+1)
        df1 = training_data[training_data['engine_no'] == i+1]
        max_cycle = max(df1['cycle'])
        df1['RUL'] = df1['cycle'].apply(lambda x: max_cycle-x)
        # cap RUL to 160 the designed lift
        df1['RUL'] = df1['RUL'].apply(lambda x: RUL_cap if x > RUL_cap else x)
        df2 = df1.drop(['engine_no'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_train = df_train.append(df3)     # dataframes under each other
    
    df_train.rename(columns={'RUL': 'Y'}, inplace=True)

    return df_train


def load_test_data():
    test_data = load_data(test_file)
    max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
    window_size = max_window_size - 2   # 31 - 2 =29
    test_data = test_data.drop(
        ['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1)  # FD001

    data_RUL = pd.read_csv(rul_file,  header=None)
    num_engine_t = data_RUL.shape[0]
    df_test = pd.DataFrame()
    for i in range(num_engine_t):
        df1 = test_data[test_data['engine_no'] == i+1]
        max_cycle = max(df1['cycle']) + data_RUL.iloc[i, 0]
        # Calculate Y (RUL)
        df1['RUL'] = max_cycle - df1['cycle']
        df1['RUL'] = df1['RUL'].apply(lambda x: RUL_cap if x > RUL_cap else x)
        # df2 = df1.drop(['engine_no', 'cycle'], axis=1)
        df2 = df1.drop(['engine_no'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_test = df_test.append(df3)

    df_test.rename(columns={'RUL': 'Y'}, inplace=True)
    
    return df_test


def normalize_data(df_train, df_test):
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_values = df_train.drop('Y', axis=1).values  # only normalize X, not y
    dim1 = len(train_values)
    dim2 = train_values[0][0].shape[0]
    dim3 = train_values[0][0].shape[1]
    train_values_2D = np.zeros((dim1*dim2, dim3))
    for in1 in range(dim1):
        train_values_2D[in1*dim2:in1*dim2+dim2][:] = train_values[in1][0]
    train_values_2D = train_values_2D.astype('float32')

    scaled_train = scaler.fit_transform(train_values_2D)
    scaled_train = train_values_2D.reshape(-1,dim2, dim3)


    test_values = df_test.drop('Y', axis=1).values  # only normalize X, not y
    dim1_test = len(test_values)
    test_values_2D = np.zeros((dim1_test*dim2, dim3))
    for in1 in range(dim1_test):
        test_values_2D[in1*dim2:in1*dim2+dim2][:] = test_values[in1][0]
    test_values_2D = test_values_2D.astype('float32')

    scaled_test = scaler.transform(test_values_2D)
    scaled_test = scaled_test.reshape(-1,dim2,dim3)

    return scaled_train, scaled_test

