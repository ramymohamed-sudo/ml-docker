


import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import csv
import os 
import sys
from load_data_2d import *     

validation_split=0.2
epochs = 1   # 10000

model_dir='../' # directory to save model history after every epoch 
model_dir_for_logs_and_h5 = model_dir+'logs&h5-models/'
if not ('logs&h5-models' in os.listdir(model_dir)):
    model_dir_for_logs_and_h5 = './scripts/logs&h5-models/'

df_train = load_train_data()
df_test = load_test_data()
train_X, test_X = normalize_data(df_train, df_test)
train_y = df_train['Y'].values.astype('float32')
test_y = df_test['Y'].values.astype('float32')



""" Save the history logs from model.fit() """
# You can achieve this functionality by creating a class which sub-classes tf.keras.callbacks.Callback 
# and use the object of that class as callback to model.fit.


class StoreModelHistory(keras.callbacks.Callback):
#   def __init__(self, name):

  def on_epoch_end(self,batch,logs=None):
    if ('lr' not in logs.keys()):
      logs.setdefault('lr',0)
      logs['lr'] = K.get_value(self.model.optimizer.lr)

    if not (f'lstm2d_with_scale_epochs_{epochs}.csv' in os.listdir(model_dir_for_logs_and_h5)):
      with open(model_dir_for_logs_and_h5+f'lstm2d_with_scale_epochs_{epochs}.csv','a') as f:
        y=csv.DictWriter(f,logs.keys())
        y.writeheader()

    with open(model_dir_for_logs_and_h5+f'lstm2d_with_scale_epochs_{epochs}.csv','a') as f:
      y=csv.DictWriter(f,logs.keys())
      y.writerow(logs)



""" 7. Train a LSTM model """
dim1 = train_X.shape[0]
dim2 = train_X.shape[1]
dim3 = train_X.shape[2]
print(f" dim1 is {dim1}\n dim2 is {dim2}\n dim3 is {dim3}")

tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(units=80, input_shape=(dim2, dim3), return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(LSTM(units=40, return_sequences=False))
model.add(Dropout(rate=0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',
                metrics=['mean_absolute_error', 'RootMeanSquaredError'])
# metrics=[keras.metrics.MeanSquaredError()]
# metrics=[keras.metrics.RootMeanSquaredError()]

# reshape input to be 3D [samples, timesteps, features]
history = model.fit(train_X.reshape(dim1, dim2, dim3),
                    train_y, epochs=epochs, batch_size=1000,
                    validation_split=validation_split, verbose=2, shuffle=True,
                    callbacks=[StoreModelHistory()])
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
size_before_reshape = train_X.reshape(dim1, dim2, dim3)
print("size before and after prediction",
        train_X.shape, size_before_reshape.shape)
yhat = model.predict(train_X.reshape(dim1, dim2, dim3))
print("yhat: ", yhat)
model.save(model_dir_for_logs_and_h5+f'lstm2d_w_scale_epochs_{epochs}.h5')
# !tar -zcvf LSTM2d_model_w_scale.tgz LSTM2d_model_w_scale.h5
# !ls -l

