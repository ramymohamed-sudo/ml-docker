

# https://www.kaggle.com/furkancitil/nasa-bearing-dataset-rul-prediction

""" This notebook contains followings:

Data Merging
Feature Extraction
Model Selection
Hyperparameter Tuning
Results on All Datasets
References
Conclusion
Additional Resources """ 

#importing necessary libraries
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn import preprocessing
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import plotly.express as px
import plotly.graph_objects as go
import optuna


#Data paths
dataset_path_1st = '../input/bearing-dataset/1st_test/1st_test'
dataset_path_2nd = '../input/bearing-dataset/2nd_test/2nd_test'
dataset_path_3rd = '../input/bearing-dataset/3rd_test/4th_test/txt'

# Test for the first file
dataset = pd.read_csv('../input/bearing-dataset/1st_test/1st_test/2003.10.22.12.06.24', sep='\t')
ax = dataset.plot(figsize = (24,6), title= "Bearing Vibration" , legend = True)
ax.set(xlabel="cycle(n)", ylabel="vibration/acceleration(g)")
plt.show()

""" Feature Extraction """
# Absolute Mean:
# Standart Deviation:
# Skewness:
# Kurtosis:
# Entropy:
# RMS:
# Peak to Peak:
# Crest Factor:
# Clearence Factor:
# Shape Factor:
# Impulse:


# Root Mean Squared Sum
def calculate_rms(df):
    result = []
    for col in df:
        r = np.sqrt((df[col]**2).sum() / len(df[col]))
        result.append(r)
    return result

# extract peak-to-peak features
def calculate_p2p(df):
    return np.array(df.max().abs() + df.min().abs())

# extract shannon entropy (cut signals to 500 bins)
def calculate_entropy(df):
    ent = []
    for col in df:
        ent.append(entropy(pd.cut(df[col], 500).value_counts()))
    return np.array(ent)
# extract clearence factor
def calculate_clearence(df):
    result = []
    for col in df:
        r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
        result.append(r)
    return result

def time_features(dataset_path, id_set=None):
    time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
    cols1 = ['B1_x','B1_y','B2_x','B2_y','B3_x','B3_y','B4_x','B4_y']
    cols2 = ['B1','B2','B3','B4']
    
    # initialize
    if id_set == 1:
        columns = [c+'_'+tf for c in cols1 for tf in time_features]
        data = pd.DataFrame(columns=columns)
    else:
        columns = [c+'_'+tf for c in cols2 for tf in time_features]
        data = pd.DataFrame(columns=columns)

        
        
    for filename in os.listdir(dataset_path):
        # read dataset
        raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
        
        # time features
        mean_abs = np.array(raw_data.abs().mean())
        std = np.array(raw_data.std())
        skew = np.array(raw_data.skew())
        kurtosis = np.array(raw_data.kurtosis())
        entropy = calculate_entropy(raw_data)
        rms = np.array(calculate_rms(raw_data))
        max_abs = np.array(raw_data.abs().max())
        p2p = calculate_p2p(raw_data)
        crest = max_abs/rms
        clearence = np.array(calculate_clearence(raw_data))
        shape = rms / mean_abs
        impulse = max_abs / mean_abs
        
        if id_set == 1:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,8), columns=[c+'_mean' for c in cols1])
            std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
            skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
            entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])
            rms = pd.DataFrame(rms.reshape(1,8), columns=[c+'_rms' for c in cols1])
            max_abs = pd.DataFrame(max_abs.reshape(1,8), columns=[c+'_max' for c in cols1])
            p2p = pd.DataFrame(p2p.reshape(1,8), columns=[c+'_p2p' for c in cols1])
            crest = pd.DataFrame(crest.reshape(1,8), columns=[c+'_crest' for c in cols1])
            clearence = pd.DataFrame(clearence.reshape(1,8), columns=[c+'_clearence' for c in cols1])
            shape = pd.DataFrame(shape.reshape(1,8), columns=[c+'_shape' for c in cols1])
            impulse = pd.DataFrame(impulse.reshape(1,8), columns=[c+'_impulse' for c in cols1])
            
        else:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,4), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])
            rms = pd.DataFrame(rms.reshape(1,4), columns=[c+'_rms' for c in cols2])
            max_abs = pd.DataFrame(max_abs.reshape(1,4), columns=[c+'_max' for c in cols2])
            p2p = pd.DataFrame(p2p.reshape(1,4), columns=[c+'_p2p' for c in cols2])
            crest = pd.DataFrame(crest.reshape(1,4), columns=[c+'_crest' for c in cols2])
            clearence = pd.DataFrame(clearence.reshape(1,4), columns=[c+'_clearence' for c in cols2])
            shape = pd.DataFrame(shape.reshape(1,4), columns=[c+'_shape' for c in cols2])
            impulse = pd.DataFrame(impulse.reshape(1,4), columns=[c+'_impulse' for c in cols2])
            
        mean_abs.index = [filename]
        std.index = [filename]
        skew.index = [filename]
        kurtosis.index = [filename]
        entropy.index = [filename]
        rms.index = [filename]
        max_abs.index = [filename]
        p2p.index = [filename]
        crest.index = [filename]
        clearence.index = [filename]
        shape.index = [filename]
        impulse.index = [filename] 
        
        # concat
        merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p,crest,clearence, shape, impulse], axis=1)
        data = data.append(merge)
        
    if id_set == 1:
        cols = [c+'_'+tf for c in cols1 for tf in time_features]
        data = data[cols]
    else:
        cols = [c+'_'+tf for c in cols2 for tf in time_features]
        data = data[cols]
        
    data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
    data = data.sort_index()
    return data


# Calling feature extraction function defined above to merge extracted features
# Saving as .csv file
set1 = time_features(dataset_path_1st, id_set=1)
set1.to_csv('set1_timefeatures.csv')

# Reading Data again
set1 = pd.read_csv("./set1_timefeatures.csv")

# Changing indexing column to time which is also name of the each file
set1 = set1.rename(columns={'Unnamed: 0':'time'})
set1.set_index('time')
set1.head()

time_features_list = ["mean","std","skew","kurtosis","entropy","rms","max","p2p", "crest", "clearence", "shape", "impulse"]
B4_x = ["B4_x_"+i for i in time_features_list]
B4_x = ['time']+B4_x
features = set1[B4_x]
#features = features.set_index("time")
features.head()

features.plot(x="time", y = "B4_x_mean")

#simple moving average SMA
ma = pd.DataFrame()
ma['B4_x_mean'] = features['B4_x_mean']
ma['SMA'] = ma['B4_x_mean'].rolling(window=5).mean()
ma['time'] = features['time']

#Cumulative Moving Average
ma['CMA'] = ma["B4_x_mean"].expanding(min_periods=10).mean()

#Exponantial Moving Average
ma['EMA'] = ma['B4_x_mean'].ewm(span=40,adjust=False).mean()

ma.plot(x="time", y= ['B4_x_mean','SMA','CMA','EMA'])

for ft in B4_x[1:]:
    col = ft+"-EMA"
    features[ft] = features[ft].ewm(span=40,adjust=False).mean()


""" PCA """
#Standardize
features_scaled = features.copy()
features_scaled = features_scaled[B4_x[1:]]
features_scaled = (features_scaled - features_scaled.mean(axis=0)) / features_scaled.std(axis=0)
features_scaled.head()

from sklearn.decomposition import PCA
pca = PCA()

X_pca = pca.fit_transform(features_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
X_pca.head()

loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=features_scaled.columns,  # and the rows are the original features
)
loadings


X_pca.plot()

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()


health_indicator = np.array(X_pca['PC1'])
degredation = pd.DataFrame(health_indicator,columns=['PC1'])
degredation['time'] = set1['time']
degredation['cycle'] = degredation.index
degredation = degredation.set_index('time')
degredation['PC1'] = degredation['PC1']-degredation['PC1'].min(axis=0)
degredation.plot(x='cycle',y='PC1')


""" RUL Prediction """
#cycle
forecast_point = 1435
thresh=6
#fitting exp function
from scipy.optimize import curve_fit

x =np.array(degredation.cycle)

y = np.array(degredation.PC1)
def exp_fit(x,a,b,c):
    y = a*np.exp(b*x)+c
    return y
fit = curve_fit(exp_fit,x[:forecast_point],y[:forecast_point],p0=[0.01,0.001,0.01])
fit = fit[0]
fit_eq = fit[0]*np.exp(fit[1]*x[:forecast_point])+fit[2]
print(fit)

fig =plt.figure()
ax =fig.subplots()
ax.scatter(x,np.log(y),color='b',s=5)
#ax.plot(x[:forecast_point],fit_eq,color='r',alpha=0.7)

plt.show()

# References
# https://www.kaggle.com/yasirabd/nasa-bearing-feature-extraction
# https://www.mathworks.com/help/predmaint/ug/signal-features.html
# https://www.mdpi.com/2076-3417/10/16/5639/htm#B13-applsci-10-05639
# Additional Resources
# https://www.youtube.com/watch?v=YtebGVx-Fxw&list=RDCMUCtYLUTtgS3k1Fg4y5tAhLbw&start_radio=1
# Cavalaglio Camargo Molano, J., Strozzi, M., Rubini, R., & Cocconcelli, M. (2019). 
# Analysis of NASA Bearing Dataset of the University of Cincinnati by Means of 
# Hjorth’s Parameters. In International Conference on Structural Engineering Dynamics 
# ICEDyn 2019.


