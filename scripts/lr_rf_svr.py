


from numpy.core.defchararray import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


df_train = load_train_data()
df_test = load_test_data()
train_X, test_X = normalize_data(df_train, df_test)
train_y = df_train['Y'].values.astype('float32')
test_y = df_test['Y'].values.astype('float32')



###############################################################
print("before train_test_split")
X_train, X_test, y_train, y_test = train_test_split(train_X,
                                                    train_y,
                                                    test_size=0.2,
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
