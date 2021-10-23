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
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

n_train = 10   # 10000


pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


start_time = datetime.datetime.now()
file_path = './CMAPSSData/'
train_file = 'train_FD001.txt'
test_file = 'test_FD001.txt'
rul_file = 'RUL_FD001.txt'
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


def load_data(data_path):
    operational_settings = ['os{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sm{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'cycle'] + operational_settings + sensor_columns
    data = pd.read_csv(data_path, sep=' ', header=None, names=cols)
    data = data.drop(cols[-5:], axis=1)
    # data['index'] = data.index
    # data.index = data['index']
    # data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')

    # print(f'Loaded data with {data.shape[0]} Recordings')    # Recordings\n{} Engines
    # print(f"Number of engines {len(data['engine_no'].unique())}")
    # print('21 Sensor Measurements and 3 Operational Settings')
    return data


if __name__ == "__main__":
    """ 1. Load training data """
    training_data = load_data(file_path+train_file)
    # print("data.head():\n", training_data.head())
    # print("data.shape:\n", training_data.shape)     # (20631, 26)
    # print("data.var():\n", training_data.drop(
    #     ['engine_no', 'cycle'], axis=1).var())
    num_engine = max(training_data['engine_no'])
    print("num_engine:", num_engine)

    """ 2. Load test data """
    test_data = load_data(file_path+test_file)
    test_data.head()
    test_data.shape
    test_data.groupby('engine_no')['cycle'].max()

    # windows size can't be great than max_window_size-2
    max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
    window_size = max_window_size - 2

    """ 3. Load RUL """
    data_RUL = pd.read_csv(file_path+rul_file,  header=None)
    # print("data_RUL.shape", data_RUL.shape)
    num_engine_t = data_RUL.shape[0]
    print("num_engine_t", num_engine_t)

    """ 4. Remove columns that are not useful for prediction """
    training_data.columns
    # Follow variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19 for FD001
    training_data = training_data.drop(
        ['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1)  # FD001
    # Follow variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19
    test_data = test_data.drop(
        ['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1)  # FD001
    training_data.head()
    training_data.tail()
    training_data.columns
    """ 5. Define window size. Convert time series to features. """
    # df is the training data
    df_train = pd.DataFrame()

    """ series to supervised TRAINING data """
    # For each engine calculate RUL. Loops thru 100 engines in train_FD001.txt
    # Change num_engine if you use other dataset
    # Call series_to_supervised to conver time series to features
    RUL_cap = 130
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
  
    # print("df_train.shape", df_train.shape)     # # (17731, 2)
    """ series to supervised TEST data """ 
    df_test = pd.DataFrame()
    # For each engine calculate RUL. Loops thru 100 engines in test_FD001.txt and RUL_FD001.txt
    # Change num_engine_t if you use other dataset

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

    # print("df_test.head()", df_test.head())
    # print("df_test.shape", df_test.shape)
    # print("df_test.columns", df_test.columns)
    df_train.rename(columns={'RUL': 'Y'}, inplace=True)
    df_test.rename(columns={'RUL': 'Y'}, inplace=True)
    

    """ 6. Normalize the features X """
    # normalize features to produce a better prediction
    train_values = df_train.drop('Y', axis=1).values  # only normalize X, not y
    dim1 = len(train_values)
    dim2 = train_values[0][0].shape[0]
    dim3 = train_values[0][0].shape[1]
    # print("train_values.shape", train_values.shape)
    
    train_values_2D = np.zeros((dim1*dim2, dim3))
    for in1 in range(dim1):
        train_values_2D[in1*dim2:in1*dim2+dim2][:] = train_values[in1][0]
    train_values_2D = train_values_2D.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_values_2D)
    scaled_train = train_values_2D.reshape(-1,dim2, dim3)
    # scaled_train = scaled_train.reshape(dim1,dim2,-1)
    print("scaled.shape", scaled_train.shape)

    test_values = df_test.drop('Y', axis=1).values  # only normalize X, not y
    dim1_test = len(test_values)
    test_values_2D = np.zeros((dim1_test*dim2, dim3))
    for in1 in range(dim1_test):
        test_values_2D[in1*dim2:in1*dim2+dim2][:] = test_values[in1][0]
    test_values_2D = test_values_2D.astype('float32')
    # normalize test features
    scaled_test = scaler.transform(test_values_2D)
    scaled_test = scaled_test.reshape(-1,dim2,dim3)
    print("scaled_t.shape", scaled_test.shape)

    # split into train and test sets
    train_X = scaled_train
    test_X = scaled_test
    # num_engine_t 100
    # scaled.shape (17731, 510)
    # scaled_t.shape (10196, 510)

    # split into input and outputs
    train_y = df_train['Y'].values.astype('float32')
    test_y = df_test['Y'].values.astype('float32')


    """ 7. Train a LSTM model """
    # for 2-D approach this value is train_X.shape[1], for 3-D approach it's train_X.shape[1]/dim2

    # clear tf cache
    tf.keras.backend.clear_session()
    # tf.random.set_seed(51) #tf 2.x
    # tf.random.set_random_seed(71) #tf 1.x
    # np.random.seed(71)
    # TF 1.x
    # For LSTM, (dim2, dim3) to be (timestamp, no_of_features)
    model = Sequential()
    model.add(LSTM(units=80, input_shape=(dim2, dim3), return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(units=40, return_sequences=False))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam',
                  metrics=['mean_absolute_error'])


    # reshape input to be 3D [samples, timesteps, features]
    history = model.fit(train_X,
                        train_y, epochs=n_train, batch_size=1000,
                        validation_split=0.1, verbose=2, shuffle=True)
    print(model.summary())

    sys.exit()


    # plot history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # 177.02965632527898
    print("min(history.history['val_loss'])", min(history.history['val_loss']))
    history.history['val_loss'].index(min(history.history['val_loss']))

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mean_absolute_error'], label='train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='test MAE')
    plt.legend()
    plt.show()

    # epoch > 150
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'][450:], label='train')
    plt.plot(history.history['val_loss'][450:], label='test')
    plt.legend()
    plt.show()

    print("min(history.history['val_loss'])", min(history.history['val_loss']))
    history.history['val_loss'].index(min(history.history['val_loss']))

    # make a prediction
    size_before_reshape = train_X.reshape(dim1, dim2, dim3)     # see dim1
    print("size before and after prediction",
          train_X.shape, size_before_reshape.shape)
    yhat = model.predict(train_X.reshape(dim1, dim2, dim3))     # see dim1
    model.save('LSTM_model_w_scale.h5')  # creates a HDF5 file 'my_model.h5'
    # !tar -zcvf LSTM_model_w_scale.tgz LSTM_model_w_scale.h5
    # !ls -l


# Epoch 1000/1000
#  16/16 -loss: 196.9570 - mean_absolute_error: 10.3632 - val_loss: 188.4589 - val_mean_absolute_error: 10.1835


sys.exit()
# reshape input to be 3D [samples, timesteps, features]

# make a prediction
yhat = model.predict(train_X.reshape(dim1, dim2, dim3))     # see dim1

rmse1 = sqrt(mean_squared_error(yhat, train_y))
print("RMSE on " + train_file + " : %.3f " % rmse1)


plt.figure(figsize=(15, 6))
plt.plot(yhat, label='predict')
plt.plot(train_y, label='true')
plt.legend()
plt.show()


plt.figure(figsize=(15, 6))
plt.grid(True)
plt.plot(yhat[-1500:], label='predict')
plt.plot(train_y[-1500:], label='true')
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
plt.grid(True)
plt.plot(yhat[0:1500], label='predict')
plt.plot(train_y[0:1500], label='true')
plt.legend()
plt.show()

""" 8. Last prediction of each engine """
# get the last prediction of each engine
ru = pd.DataFrame(data.groupby('engine_no')[
                  'cycle'].max() - window_size).reset_index()
ru.columns = ['id', 'last_prediction']

true_y = [0]*num_engine
#true_y = list(data_r[0])

pred_y = []
sm = -1
for j in range(len(true_y)):
    sm = sm + ru.iloc[j, 1]  # the index of last predition in yh
    pred_y.append(yhat[sm])

# calculate RMSE
rmse2 = sqrt(mean_squared_error(pred_y, true_y))

print('RMSE for last prediction of each engine in ' +
      train_file + ' : %.3f' % rmse2)

plt.figure(figsize=(15, 6))
plt.plot(pred_y, label='predict')
plt.plot(true_y, label='true')
plt.legend()
plt.title("Last predictions for %3d engines " % len(true_y))
plt.show()

""" 9. Use the model on the un-seen dataset df_t to make predictions """
tt_X, tt_y = scaled_t[:, :-1], scaled_t[:, -1]
print(tt_X.shape, tt_y.shape)


# reshape input to be 3D [samples, timesteps, features]

# make a prediction
yh = model.predict(test_X.reshape(test_X.shape[0], dim2, dim3))
rmse3 = sqrt(mean_squared_error(yh, test_y))
print("Test RMSE on " + test_file + " : %.3f" % rmse3)


plt.figure(figsize=(15, 6))
plt.grid(True)
plt.plot(yh[0:1500], label='predict')
plt.plot(test_y[0:1500], label='true')
plt.legend()
plt.title("Prediction for first few engines")
plt.show()


plt.figure(figsize=(15, 6))
plt.grid(True)
plt.plot(yh[-1500:], label='predict')
plt.plot(test_y[-1500:], label='true')
plt.legend()
plt.title("Prediction for last few engines")
plt.show()

# See the error trend in terms of RUL number
num_prediction = [0]*7
rmse = [0]*7

for i in range(7):
    low_b = i*20
    high_b = (i+1)*20
    ct = 0
    sq_sum = 0
    for j in range(len(test_y)):
        if low_b < test_y[j] <= high_b:
            ct += 1
            sq_sum += (test_y[j] - yh[j][0]) ** 2
    rmse[i] = (sq_sum/ct)**(1/2)
    num_prediction[i] = ct
print(num_prediction)
print(rmse)

labels = ['RUL0-20', 'RUL20-40', 'RUL40-60',
          'RUL60-80', 'RUL80-100', 'RUL100-120', 'RUL120-140']

dfb = pd.DataFrame(list(zip(num_prediction, rmse)), columns=[
                   'num_prediction', 'RMSE'], index=labels)
dfb.plot(kind='bar', figsize=(10, 10), grid=True, subplots=True, sharex=True)

data_r.shape
data_t.shape
ru.tail
len(yh)

# get the last prediction of each engine
ru = pd.DataFrame(data_t.groupby('engine_no')[
                  'cycle'].max() - window_size).reset_index()
ru.columns = ['id', 'last_prediction']

true_y = list(data_r[0])

pred_y = []
sm = -1
for j in range(len(true_y)):
    sm = sm + ru.iloc[j, 1]  # the index of last predition in yh
    pred_y.append(yh[sm])

# calculate RMSE
rmse4 = sqrt(mean_squared_error(pred_y, true_y))

print('Test RMSE for last prediction of each engine : %.3f' % rmse4)

plt.figure(figsize=(15, 6))
plt.plot(pred_y, label='predict')
plt.plot(true_y, label='true')
plt.legend()
plt.title("Last predictions for %3d engines " % len(true_y))
plt.show()

err = []
for k in range(len(true_y)):
    err.append(true_y[k] - pred_y[k][0])

plt.figure(figsize=(15, 6))
plt.plot(err, label='error')
plt.legend()
plt.title("Last predictions error for %3d engines " % len(true_y))
plt.show()

""" 11. Score one engine a time """


def one_engine(engine_no):
    df_one = pd.DataFrame()
    i = engine_no  # engine no. 0-99

    df1 = data_t[data_t['engine_no'] == i+1]
    max_cycle = max(df1['cycle']) + data_r.iloc[i, 0]
    # Calculate Y (RUL)
    df1['RUL'] = max_cycle - df1['cycle']
    df1['RUL'] = df1['RUL'].apply(lambda x: RUL_cap if x > RUL_cap else x)
    #df2 = df1.drop(['engine_no', 'cycle'], axis=1)
    df2 = df1.drop(['engine_no'], axis=1)
    df_one = series_to_supervised(df2, window_size, 1)

    df_one.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    # Drop all RUL columns since they are unknown when doing prediction
    for co in df_one.columns:
        if co.startswith('RUL'):
            df_one.drop([co], axis=1, inplace=True)

    values_o = df_one.drop('Y', axis=1).values
    values_o = values_o.astype('float32')
    scaled_o = scaler.transform(values_o)

    t1_X, t1_y = scaled_o, df_one['Y'].values.astype('float32')

    # make a prediction
    yh1 = model.predict(t1_X.reshape((t1_X.shape[0], dim2, dim3)))
    rmse5 = sqrt(mean_squared_error(yh1, t1_y))
    print('Test RMSE on engine no.%3d in ' %
          (i+1) + test_file + ' : %.3f' % (rmse5))

    plt.figure(figsize=(10, 6))
    plt.plot(yh1, label='predict')
    plt.plot(t1_y, label='true')
    plt.legend()
    plt.title("Prediction for engines no. " + str(i+1))
    plt.show()
    return rmse5


# engine 100 - good prediction
rmse5 = one_engine(99)
