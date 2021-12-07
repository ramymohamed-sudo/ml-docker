


from numpy.core.defchararray import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# Multi-layer neural network MLP

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
from sklearn.utils import shuffle
from load_data import *

test_size=0.2

model_dir_for_logs_and_h5 = model_dir+'logs&h5-models/'
if not ('logs&h5-models' in os.listdir(model_dir)):
    model_dir_for_logs_and_h5 = './scripts/logs&h5-models/'


df_train = load_train_data()
df_test = load_test_data()
train_X, test_X = normalize_data(df_train, df_test)
train_y = df_train['Y'].values.astype('float32')
test_y = df_test['Y'].values.astype('float32')



###############################################################
print("before train_test_split")
X_train, X_test, y_train, y_test = train_test_split(train_X,
                                                    train_y,
                                                    test_size=test_size,
                                                    random_state=42,
                                                    shuffle=True)
print("shape of X_train", X_train.shape)
print("shape of X_test", X_test.shape)
print("shape of y_train", y_train.shape)
print("shape of y_test", y_test.shape)


""" Linear Regression """
linear_model_reg = LinearRegression()

""" Random Forest Model """
rf_model_reg = RandomForestRegressor(random_state=42)

""" SVM Model """
svr_model_reg = SVR(kernel='rbf')

""" Pipeline """
reg_pipe = Pipeline([("regression", rf_model_reg)])
#  for NLP Pipeline([("TF-IDF", tf_idf_model), ("regression", rf_model_reg)])

model1_reg = {"regression": [linear_model_reg]}

model2_reg = {"regression": [rf_model_reg],
              "regression__n_estimators": [10, 100],
              "regression__max_features": [1, 3]}

model3_reg = {"regression": [svr_model_reg],
              "regression__C": [1, 5, 10],
              "regression__gamma": ('auto', 'scale'),
              'regression__kernel': ('linear', 'poly', 'rbf', 'sigmoid')}

search_space_reg = [model1_reg, model2_reg, model3_reg]
clf_reg = GridSearchCV(reg_pipe, search_space_reg, cv=5, n_jobs=-1)

best = clf_reg.fit(X_train, y_train)
print("best.best_params_", best.best_params_)
print("best.best_score_", best.best_score_)


means = clf_reg.cv_results_['mean_test_score']
print("means", means)
for mean, params in zip(means, clf_reg.cv_results_['params']):
    print("% 0.3f for % r " % (mean, params))

y_pred = clf_reg.predict(X_test)

mean_abs_err = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')

mean_sqrd_err = mean_squared_error(y_test, y_pred, multioutput='uniform_average', squared=True)

print(f"Final mean_absolute_error is {mean_abs_err}")
print(f"Final mean_squared_error is {mean_sqrd_err}")



# Bar plot of MAE/RMSE measures of our proposed CBLSTMs without dropout, CLSTM and CBLSTM under
# three different datasets: C1, C4 and C6, respectively.
# Finally, three tool life tests named C1, C4 and C6 were selected as our dataset. Each test contains 315 data samples, while each data sample has a corresponding flank wear. 
# For training/testing splitting, a three-fold setting is adopted such that two tests are used as the training domain and the other one is used as the testing domain. For example, when C4 and C6 are used as the training datasets, C1 will be adopted as the testing dataset. This splitting is denoted as c1.
# Our task is defined as the prediction of tool wear depth based on the sensory input.
# Total wear is plotted versus the number of cuts


# Table 3. MAE for compared methods on these three datasets. Bold face indicates the best performances.
# RNN, Recurrent Neural Network.

# Table 4. RMSE for compared methods on these three datasets. Bold face indicates the best performance
