


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


pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = datetime.datetime.now()

RUL_cap = 130


model_dir='../' 
file_path = model_dir + 'CMAPSSData/'
if not ('CMAPSSData' in os.listdir(model_dir)):
    file_path = './scripts/CMAPSSData/'


train_file = file_path+'train_FD001.txt'
test_file = file_path+'test_FD001.txt'
rul_file = file_path+'RUL_FD001.txt'


# convert series to supervised learning
def series_to_supervised(data, window_size=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    # print("n_vars", n_vars)  # 18
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(window_size, 0, -1):        # window_size=29 for engine no 1
        # print(f"i is {i}")
        cols.append(df.shift(i))
        # print("cols", cols)
        names += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
        # print("names", names)

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(df.columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_data(data_path):
    operational_settings = ['os{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sm{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'cycle'] + operational_settings + sensor_columns
    data = pd.read_csv(data_path, sep=' ', header=None, names=cols)
    data = data.drop(cols[-5:], axis=1)
    # cols = ['engine_no', 'cycle', 'os1', 'os2', 'os3', 'sm1',....., 'sm21']
    # data['index'] = data.index
    # data.index = data['index']
    # data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')
    # print(f'Loaded data with {data.shape[0]} Recordings')    # Recordings\n{} Engines
    # print(f"Number of engines {len(data['engine_no'].unique())}")
    # print('21 Sensor Measurements and 3 Operational Settings')
    return data


def load_train_data():
    """ 1. Load training data """
    training_data = load_data(train_file)
    # print("data.head():\n", training_data.head())
    # print("data.shape:\n", training_data.shape)
    # print("data.var():\n", training_data.drop(
    #     ['engine_no', 'cycle'], axis=1).var())
    num_engine = max(training_data['engine_no'])
    print(f"num_engine is {num_engine}")

    """ 2. Load test data to get window_size"""
    test_data = load_data(test_file)
    
    max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
    window_size = max_window_size - 2   # 31 - 2 =29

    """ 3. Remove columns that are not useful for prediction """
    # Following variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19 for FD001
    training_data = training_data.drop(
        ['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1)  # FD001
    
    """ 4. Convert time series to features. """
    # df is the training data
    df_train = pd.DataFrame()
    num_engine = 6
    #training_data.groupby('engine_no')
    # x = training_data.value_counts('engine_no').sort_index()
    # x = training_data.engine_no.value_counts().sort_index()
    # print("x\n", x)
    

    for i in range(num_engine):
        df1 = training_data[training_data['engine_no'] == i+1]

        if i == 4:
            data_of_sensor_05 = training_data[training_data['engine_no']== 5]
            print(data_of_sensor_05.head())
            data_of_sensor_05.plot("cycle", "sm2",figsize=(5,5))
            plt.show()
            sys.exit()




        max_cycle = max(df1['cycle'])
        print(f"max_cycle for engine no. {i+1} is {max_cycle}")
        print("df1.shape", df1.shape)   # df1.shape (192, 18)
        # print("df1 max_cycle", max_cycle)   # df1 max_cycle 192
        # Calculate Y (RUL)
        # df1['RUL'] = max_cycle - df1['cycle']
        df1['RUL'] = df1['cycle'].apply(lambda x: max_cycle-x)
        df1['RUL'] = df1['RUL'].apply(lambda x: RUL_cap if x > RUL_cap else x)
        df2 = df1.drop(['engine_no'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_train = df_train.append(df3)     # dataframes under each other

    df_train.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    for col in df_train.columns:
        if col.startswith('RUL'):
            df_train.drop([col], axis=1, inplace=True)

    return df_train, data_of_sensor_05


df_train, data_of_sensor_05 = load_train_data()

print("data_of_sensor_05.head()\n", data_of_sensor_05.head())
print("data_of_sensor_05.head()\n", data_of_sensor_05)
sys.exit()

def load_test_data():
    test_data = load_data(test_file)
    max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
    window_size = max_window_size - 2   # 31 - 2 =29
    # Follow variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19
    test_data = test_data.drop(
        ['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1)  # FD001
    
    """ Load RUL """
    data_RUL = pd.read_csv(rul_file,  header=None)
    num_engine_t = data_RUL.shape[0]

    # df_t is the testing data
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
    
    df_test.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    for col in df_test.columns:
        if col.startswith('RUL'):
            df_test.drop([col], axis=1, inplace=True)
    
    return df_test


def normalize_data(df_train, df_test):
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_values = df_train.drop('Y', axis=1).values  # only normalize X, not y
    train_values = train_values.astype('float32')
    scaled_train = scaler.fit_transform(train_values)

    values_test = df_test.drop('Y', axis=1).values  # only normalize X, not y
    values_test = values_test.astype('float32')
    scaled_test = scaler.transform(values_test)

    return scaled_train, scaled_test

