


import tensorflow as tf
# import keras
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from load_data import *


epochs = 10000   # 10000


df_train = load_train_data()
df_test = load_test_data()
train_X, test_X = normalize_data(df_train, df_test)
train_y = df_train['Y'].values.astype('float32')
test_y = df_test['Y'].values.astype('float32')


""" 7. Train a LSTM model """
dim1 = train_X.shape[0]
dim2 = 1    # for 2-D approach this value is 1
dim3 = int(train_X.shape[1]/dim2)

tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(units=80, input_shape=(dim2, dim3), return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=40, return_sequences=False))
model.add(Dropout(rate=0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',
                metrics=['mean_absolute_error'])

# reshape input to be 3D [samples, timesteps, features]
history = model.fit(train_X.reshape(dim1, dim2, dim3),
                    train_y, epochs=epochs, batch_size=1000,
                    validation_split=0.1, verbose=2, shuffle=True)
print(model.summary())

sys.exit()
# Epoch 10/10


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
size_before_reshape = train_X.reshape(dim1, dim2, dim3)
print("size before and after prediction",
        train_X.shape, size_before_reshape.shape)
yhat = model.predict(train_X.reshape(dim1, dim2, dim3))
model.save(f'kera_lstm_w_scale_epochs_{epochs}.h5')  # creates a HDF5 file 'my_model.h5'
# !tar -zcvf LSTM_model_w_scale.tgz LSTM_model_w_scale.h5
# !ls -l


sys.exit()
# reshape input to be 3D [samples, timesteps, features]

# make a prediction
yhat = model.predict(train_X.reshape(dim1, dim2, dim3))

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
