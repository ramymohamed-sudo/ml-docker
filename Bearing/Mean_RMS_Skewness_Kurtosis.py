
# https://www.kaggle.com/yasirabd/rul-nasa-bearing-mean-rms-skewness-kurtosis

# !pip install hmmlearn -q

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, random

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set_style('whitegrid')

from hmmlearn import hmm


""" Description Dataset """


""" Load Dataset """
set1 = pd.read_csv('../input/nasa-bearing-time-features/set1_timefeatures.csv')
set1.columns = ['date'] + list(set1.columns[1:])
set1['date'] = pd.to_datetime(set1['date'])

set2 = pd.read_csv('../input/nasa-bearing-time-features/set2_timefeatures.csv')
set2.columns = ['date'] + list(set2.columns[1:])
set2['date'] = pd.to_datetime(set2['date'])

set3 = pd.read_csv('../input/nasa-bearing-time-features/set3_timefeatures.csv')
set3.columns = ['date'] + list(set3.columns[1:])
set3['date'] = pd.to_datetime(set3['date'])

# set date as index
set1 = set1.set_index('date')
set2 = set2.set_index('date')
set3 = set3.set_index('date')

set1.head()



""" merge a and b from bearing 1-4 """ 
set1['B1_mean'] = (set1['B1_a_mean'] + set1['B1_b_mean'])/2
set1['B1_std'] = (set1['B1_a_std'] + set1['B1_b_std'])/2
set1['B1_skew'] = (set1['B1_a_skew'] + set1['B1_b_skew'])/2
set1['B1_kurtosis'] = (set1['B1_a_kurtosis'] + set1['B1_b_kurtosis'])/2
set1['B1_entropy'] = (set1['B1_a_entropy'] + set1['B1_b_entropy'])/2
set1['B1_rms'] = (set1['B1_a_rms'] + set1['B1_b_rms'])/2
set1['B1_max'] = (set1['B1_a_max'] + set1['B1_b_max'])/2
set1['B1_p2p'] = (set1['B1_a_p2p'] + set1['B1_b_p2p'])/2

set1['B2_mean'] = (set1['B2_a_mean'] + set1['B2_b_mean'])/2
set1['B2_std'] = (set1['B2_a_std'] + set1['B2_b_std'])/2
set1['B2_skew'] = (set1['B2_a_skew'] + set1['B2_b_skew'])/2
set1['B2_kurtosis'] = (set1['B2_a_kurtosis'] + set1['B2_b_kurtosis'])/2
set1['B2_entropy'] = (set1['B2_a_entropy'] + set1['B2_b_entropy'])/2
set1['B2_rms'] = (set1['B2_a_rms'] + set1['B2_b_rms'])/2
set1['B2_max'] = (set1['B2_a_max'] + set1['B2_b_max'])/2
set1['B2_p2p'] = (set1['B2_a_p2p'] + set1['B2_b_p2p'])/2

set1['B3_mean'] = (set1['B3_a_mean'] + set1['B3_b_mean'])/2
set1['B3_std'] = (set1['B3_a_std'] + set1['B3_b_std'])/2
set1['B3_skew'] = (set1['B3_a_skew'] + set1['B3_b_skew'])/2
set1['B3_kurtosis'] = (set1['B3_a_kurtosis'] + set1['B3_b_kurtosis'])/2
set1['B3_entropy'] = (set1['B3_a_entropy'] + set1['B3_b_entropy'])/2
set1['B3_rms'] = (set1['B3_a_rms'] + set1['B3_b_rms'])/2
set1['B3_max'] = (set1['B3_a_max'] + set1['B3_b_max'])/2
set1['B3_p2p'] = (set1['B3_a_p2p'] + set1['B3_b_p2p'])/2

set1['B4_mean'] = (set1['B4_a_mean'] + set1['B4_b_mean'])/2
set1['B4_std'] = (set1['B4_a_std'] + set1['B4_b_std'])/2
set1['B4_skew'] = (set1['B4_a_skew'] + set1['B4_b_skew'])/2
set1['B4_kurtosis'] = (set1['B4_a_kurtosis'] + set1['B4_b_kurtosis'])/2
set1['B4_entropy'] = (set1['B4_a_entropy'] + set1['B4_b_entropy'])/2
set1['B4_rms'] = (set1['B4_a_rms'] + set1['B4_b_rms'])/2
set1['B4_max'] = (set1['B4_a_max'] + set1['B4_b_max'])/2
set1['B4_p2p'] = (set1['B4_a_p2p'] + set1['B4_b_p2p'])/2

set1 = set1[['B1_mean','B1_std','B1_skew','B1_kurtosis','B1_entropy','B1_rms','B1_max','B1_p2p',
             'B2_mean','B2_std','B2_skew','B2_kurtosis','B2_entropy','B2_rms','B2_max','B2_p2p',
             'B3_mean','B3_std','B3_skew','B3_kurtosis','B3_entropy','B3_rms','B3_max','B3_p2p',
             'B4_mean','B4_std','B4_skew','B4_kurtosis','B4_entropy','B4_rms','B4_max','B4_p2p']]
set1.head()


""" Slice Features (Mean, RMS, Skewness, Kurtosis) """
cols = ['B1_mean','B1_rms','B1_skew','B1_kurtosis', 
        'B2_mean','B2_rms','B2_skew','B2_kurtosis',
        'B3_mean','B3_rms','B3_skew','B3_kurtosis',
        'B4_mean','B4_rms','B4_skew','B4_kurtosis',]
set1 = set1[cols]
set2 = set2[cols]
set3 = set3[cols]

# statistics
set1.describe().T

# statistics
set2.describe().T

# statistics
set3.describe().T

def plot_features(df):
    fig, axes = plt.subplots(4, 1, figsize=(15, 5*4))
    
    axes[0].plot(df['B1_mean'])
    axes[0].plot(df['B2_mean'])   
    axes[0].plot(df['B3_mean'])
    axes[0].plot(df['B4_mean'])
    axes[0].legend(['B1','B2','B3','B4'])
    axes[0].set_title('Mean')
    
    axes[1].plot(df['B1_rms'])
    axes[1].plot(df['B2_rms'])   
    axes[1].plot(df['B3_rms'])
    axes[1].plot(df['B4_rms'])
    axes[1].legend(['B1','B2','B3','B4'])
    axes[1].set_title('RMS')
    
    axes[2].plot(df['B1_skew'])
    axes[2].plot(df['B2_skew'])   
    axes[2].plot(df['B3_skew'])
    axes[2].plot(df['B4_skew'])
    axes[2].legend(['B1','B2','B3','B4'])
    axes[2].set_title('Skewness')
    
    axes[3].plot(df['B1_kurtosis'])
    axes[3].plot(df['B2_kurtosis'])   
    axes[3].plot(df['B3_kurtosis'])
    axes[3].plot(df['B4_kurtosis'])
    axes[3].legend(['B1','B2','B3','B4'])
    axes[3].set_title('Kurtosis')


# plot set 1
plot_features(set1)

# plot set 2
plot_features(set2)

# plot set 3
plot_features(set3)

""" Scaling """
def slice_columns(columns, target='B1'):
    if target == 'B1':
        return columns[0:4]
    elif target == 'B2':
        return columns[4:8]
    elif target == 'B3':
        return columns[8:12]
    elif target == 'B4':
        return columns[12:]



# with scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

set1_scaled = set1.copy()
set2_scaled = set2.copy()
set3_scaled = set3.copy()

ss_l = []
minmax_l = []

# scaling set test 1
for bear in ['B1','B2','B3','B4']:
    col_features = slice_columns(set1.columns, target=bear)
    scaler = StandardScaler()
#     scaler = MinMaxScaler(feature_range=(-2,2))
    set1_scaled[col_features] = scaler.fit_transform(set1[col_features])
    ss_l.append(scaler)
#     minmax_l.append(scaler)
    
# scaling set test 2
for bear in ['B1','B2','B3','B4']:
    col_features = slice_columns(set2.columns, target=bear)
    scaler = StandardScaler()
#     scaler = MinMaxScaler(feature_range=(-2,2))
    set2_scaled[col_features] = scaler.fit_transform(set2[col_features])
    ss_l.append(scaler)
#     minmax_l.append(scaler)

# scaling set test 3, except bearing 3
for bear in ['B1','B2','B4']:
    col_features = slice_columns(set3.columns, target=bear)
    scaler = StandardScaler()
#     scaler = MinMaxScaler(feature_range=(-2,2))
    set3_scaled[col_features] = scaler.fit_transform(set3[col_features])#.round(4)
    ss_l.append(scaler)
#     minmax_l.append(scaler)


# before and after scaling
fig, axes = plt.subplots(1, 2, figsize=(14,5), dpi=80)
axes[0].plot(set1['B1_mean'])
axes[1].plot(set1_scaled['B1_mean'])

axes[0].set_title('Before scaling (Mean)')
axes[1].set_title('After scaling (Mean)')

""" Prepare data """

""" GMMHMM """
def flip_transmat(tm, ix_sort):
    tm_ = tm.copy()
    for i,ix in enumerate(ix_sort):
        tm_[i, :] = tm[ix[0], :]
    tm__ = tm_.copy()
    for i,ix in enumerate(ix_sort):
        tm__[:, i] = tm_[:,ix[0]]
    return tm__


def random_transitions(n_states) -> np.ndarray:
    """Sets the transition matrix as random (random probability of transitioning
    to all other possible states from each state) by sampling probabilities
    from a Dirichlet distribution, according to the topology.
    Returns
    -------
    transitions: :class:`numpy:numpy.ndarray` (float)
        The random transition matrix of shape `(n_states, n_states)`.
    """
    transitions = np.zeros((n_states, n_states))
    for i, row in enumerate(transitions):
        row[i:] = np.random.dirichlet(np.ones(n_states - i))
    return transitions

def create_gmmhmm():
    startprob = np.array([1., 0., 0.], dtype=np.float64)
    transmat = np.array([[0.9995, 0.0005,  0.],
                         [0.,     0.9998,  0.0002],
                         [0.,     0.,      1.0]], dtype=np.float64)
#     transmat = np.array([[0.38903512, 0.28715641, 0.32380848],
#                          [0.        , 0.56796488, 0.43203512],
#                          [0.        , 0.        , 1.        ]], dtype=np.float64)
#     transmat = random_transitions(n_states=3)
    
    model = hmm.GMMHMM(n_components=3, 
                       n_mix=2, 
                       covariance_type="tied", 
                       n_iter=1000, 
                       tol=1e-6,
#                        startprob_prior=startprob,
#                        transmat_prior=transmat,
#                        init_params='cmw',
#                        params='stcmw',
                       random_state=SEED,
                       verbose=False)
    model.n_features = 4
    model.startprob_ = startprob
    model.transmat_ = transmat
    
    return model

def fit_gmmhmm(model, data):
    
    # train model
    model.fit(data)

    # clasify each observation as state (0, 1, 2)
    hidden_states = model.predict(data)
    
    # get parameters of GMMHMM
    startprob_ = model.startprob_
    means_ = model.means_
    transmat_ = model.transmat_
    covars_ = model.covars_
    weights_ = model.weights_
    
    # reorganize by mean, so the the order of the states from lower to higher
    ix_sort = np.argsort(np.array([[np.mean(m)] for m in means_]), axis=0)
    
    hidden_states = np.array([ix_sort[st][0] for st in hidden_states])
    startprob = np.array([startprob_[ix][0] for ix in ix_sort])
    means = np.array([means_[ix][0] for ix in ix_sort])
    transmat = flip_transmat(transmat_, ix_sort)
    covars = np.array([covars_[ix][0] for ix in ix_sort])
    weights = np.array([weights_[ix][0] for ix in ix_sort])
    
    model.startprob_ = startprob
    model.means_ = means
    model.transmat_ = transmat
    model.covars_ = covars
    model.weights_ = weights
    
    # logprob
    logprob = model.score(data)
    
    return ix_sort, logprob, model, hidden_states


def plot_rms_and_state(rms, state):
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax2 = ax1.twinx()
    ax1.plot(rms)
    ax1.set_ylabel('RMS')
    ax1.set_xlabel('time (minutes)')
    ax2.plot(state, color='red')

    ax2.set_yticks(range(0,3))
    ax2.set_ylabel('state', rotation=270, labelpad=20)
    plt.show()

# example on bearing 1 in the test 1

# initialize gmmhmm
gmmhmm = create_gmmhmm()

# setup data
col_features = slice_columns(set1_scaled.columns, target='B1')
data = set1_scaled[col_features]

# train gmmhmm
ix_sort, logprob, model, hidden_states = fit_gmmhmm(gmmhmm, data)

print(f'Log probability: {np.around(logprob, decimals=4)}')
print(f'Start probability: {np.around(model.startprob_, decimals=4)}')
print(f'Means:\n{np.around(model.means_, decimals=4)}')
print(f'Transition Matrix:\n{np.around(model.transmat_, decimals=3)}')
print(f'Covariance matrix:\n{np.around(model.covars_, decimals=4)}')
print(f'Weights:\n{np.around(model.weights_, decimals=4)}')
print(ix_sort)

# plot RMS and state
plot_rms_and_state(data[data.columns[1]].values, hidden_states)

%%time
# model GMMHMM
model_gmmhmm = []

# train set test 1
for bear in ['B1','B2','B3','B4']:
    gmmhmm = create_gmmhmm()  # create model
    col_features = slice_columns(set1_scaled.columns, target=bear)
    ix_sort, logprob, model, hidden_states = fit_gmmhmm(gmmhmm, set1_scaled[col_features])  # train
    
    # append model into list
    model_gmmhmm.append((ix_sort, logprob, model, hidden_states))
    
# train set test 2
for bear in ['B1','B2','B3','B4']:
    gmmhmm = create_gmmhmm()  # create model
    col_features = slice_columns(set2_scaled.columns, target=bear)
    ix_sort, logprob, model, hidden_states = fit_gmmhmm(gmmhmm, set2_scaled[col_features])  # train
    
    # append model into list
    model_gmmhmm.append((ix_sort, logprob, model, hidden_states))
    
# train set test 3, except bearing 3
for bear in ['B1','B2','B4']:
    gmmhmm = create_gmmhmm()  # create model
    col_features = slice_columns(set3_scaled.columns, target=bear)
    ix_sort, logprob, model, hidden_states = fit_gmmhmm(gmmhmm, set3_scaled[col_features])  # train
    
    # append model into list
    model_gmmhmm.append((ix_sort, logprob, model, hidden_states))



# get rms data for each bearing test, except set test 3 bearing 3 (S3_B3) because test set
rms_data = []
for bear in ['B1','B2','B3','B4']:
    col_features = slice_columns(set1_scaled.columns, target=bear)
    data = set1[col_features]
    rms = data[data.columns[1]]
    rms_data.append(rms)
for bear in ['B1','B2','B3','B4']:
    col_features = slice_columns(set2_scaled.columns, target=bear)
    data = set2[col_features]
    rms = data[data.columns[1]]
    rms_data.append(rms)
for bear in ['B1','B2','B4']:
    col_features = slice_columns(set3_scaled.columns, target=bear)
    data = set3[col_features]
    rms = data[data.columns[1]]
    rms_data.append(rms)

# sequence data for training
model_data = ['S1_B1','S1_B2','S1_B3','S1_B4',
              'S2_B1','S2_B2','S2_B3','S2_B4',
              'S3_B1','S3_B2','S3_B4']

for (ix_sort,logprob,model,hidden_states),data,rms in zip(model_gmmhmm, model_data, rms_data):
    print(f'Sequence data: {data}')
    print(f'Logprob\n {np.around(logprob, decimals=4)}')
    print(f'Start Probability {np.around(model.startprob_, decimals=4)}')
    print(f'Means\n {np.around(model.means_, decimals=4)}')
    print(f'Transition Matrix\n {np.around(model.transmat_, decimals=3)}')
    print(f'Mixture Weights\n {np.around(model.weights_, decimals=4)}')
    print(f'Covariance matrix:\n{np.around(model.covars_, decimals=4)}')

    # plot RMS and state
    plot_rms_and_state(rms.values, hidden_states)
    print()


""" Testing (S3_B3) """
# logprob and plot decode state on S3_B3
logprob_l = []
i = 0
for (ix,logprob,model,hidden_states),data in zip(model_gmmhmm, model_data):
    # select features
    col_features = slice_columns(set3.columns, target='B3')
    
    # scaling
    S3_B3 = ss_l[i].transform(set3[col_features])
    i += 1
    
    # calculate logprob on model sample
    logprob = model.score(S3_B3)
    pred = model.predict(S3_B3)
    
    logprob_l.append(logprob)

    print(f'GMMHMM model from {data} got log probability on S3_B3: {logprob}')
    rms = set3[col_features]['B3_rms']
    
    # plot RMS and decode state
    plot_rms_and_state(rms.values, pred)
    print()

# example exponential on negative log probability
np.exp(-100), np.exp(-150)

# index best model
i = 4

# select best model
ix, logprob, model, hidden_states = model_gmmhmm[i]

# transform data
S3_B3 = ss_l[i].transform(set3[col_features])
model.score(S3_B3)

# https://www.researchgate.net/publication/224177188_A_Mixture_of_Gaussians_Hidden_Markov_Model_for_failure_diagnostic_and_prognostic

def calculate_mean(durations):
    return np.mean(durations)

def calculate_std(durations):
    return np.std(durations)

import math

D_S1 = [0.458, 0.286]
D_S2 = [0.393, 0.334]
D_S3 = [0.357, 0.328]

print(f"S1 = [mean(S1), std(S1)] = [{math.floor(calculate_mean(D_S1)*10000)}, {math.floor(calculate_std(D_S1)*10000)}]")
print(f"S2 = [mean(S2), std(S2)] = [{math.floor(calculate_mean(D_S2)*10000)}, {math.floor(calculate_std(D_S2)*10000)}]")
print(f"S3 = [mean(S3), std(S3)] = [{math.floor(calculate_mean(D_S3)*10000)}, {math.ceil(calculate_std(D_S3)*10000)}]")


# The path estimation of the example on set test 1 bearing 1 above like this
# S2 -> S1 -> S2 -> S3 -> S1 -> S3

result = {'S1': {'mean': int(calculate_mean(D_S1)*10000),
                 'std': int(calculate_std(D_S1)*10000)},
          'S2': {'mean': int(calculate_mean(D_S2)*10000),
                 'std': int(calculate_std(D_S2)*10000)},
          'S3': {'mean': int(calculate_mean(D_S3)*10000),
                 'std': math.ceil(calculate_std(D_S3)*10000)}
         }
path = ['S2','S1','S2','S3','S1','S3']

result, path

def calculate_rul(path, result, conf=0.95):
    rul = []
    
    cum_rul = 0
    for p in path:
        if p == 'S1':
            rul_upper = result.get('S1').get('mean') + conf * result.get('S1').get('std') + cum_rul
            rul_mean = result.get('S1').get('mean') + cum_rul
            rul_lower = result.get('S1').get('mean') - conf * result.get('S1').get('std') + cum_rul
            
            cum_rul += result.get('S1').get('mean')
        elif p == 'S2':
            rul_upper = result.get('S2').get('mean') + conf * result.get('S2').get('std') + cum_rul
            rul_mean = result.get('S2').get('mean') + cum_rul
            rul_lower = result.get('S2').get('mean') - conf * result.get('S2').get('std') + cum_rul
            
            cum_rul += result.get('S2').get('mean')
        elif p == 'S3':
            rul_upper = result.get('S3').get('mean') + conf * result.get('S3').get('std') + cum_rul
            rul_mean = result.get('S3').get('mean') + cum_rul
            rul_lower = result.get('S3').get('mean') - conf * result.get('S3').get('std') + cum_rul
            
            cum_rul += result.get('S3').get('mean')
        
        rul.append((rul_upper, rul_mean, rul_lower))
    return rul


calculate_rul(path, result)
range_index = [(0, 3930), (3930, 8510), (8510, 11850), (11850, 15420), (15420, 18280), (18280, 21560)]

df_visual = pd.DataFrame()
df_visual['index'] = [i for i in range(0,21560,10)]
df_visual['state'] = [1]*393 + [0]*458 + [1]*334 + [2]*357 + [0]*286 + [2]*328
df_visual['rul_upper'] = [3915.25]*393 + [8172.0]*458 + [11270.25]*334 + [14552.75]*357 + [18952.0]*286 + [21697.75]*328
df_visual['rul_mean'] = [3635]*393 + [7355]*458 + [10990]*334 + [14415]*357 + [18135]*286 + [21560]*328
df_visual['rul_lower'] = [3354.75]*393 + [6538]*458 + [10709.75]*334 + [14277.25]*357 + [17318]*286 + [21422.25]*328
df_visual['rul_error_upper'] = (21560 - df_visual['rul_upper'])/21560 * 100
df_visual['rul_error_mean'] = (21560 - df_visual['rul_mean'])/21560 * 100
df_visual['rul_error_lower'] = (21560 - df_visual['rul_lower'])/21560 * 100

df_visual


# hidden state
plt.figure(figsize=(12,5), dpi=80)
plt.plot(df_visual['index'], df_visual['state'])
plt.yticks(range(0,3))

# RUL estimation
plt.figure(figsize=(12,8), dpi=80)
plt.plot(df_visual['index'], [21560]*2156, color='r', linestyle='--')
plt.plot(df_visual['index'], df_visual['rul_upper'], linestyle='--')
plt.plot(df_visual['index'], df_visual['rul_mean'])
plt.plot(df_visual['index'], df_visual['rul_lower'], linestyle='--')

plt.legend(['Real','RUL Upper', 'RUL Mean', 'RUL Lower'])
plt.ylabel('Failure time (min)')
plt.xlabel('Current time (min)')


# RUL error associated
plt.figure(figsize=(12,8), dpi=80)
plt.plot(df_visual['index'], df_visual['rul_error_upper'], linestyle='--')
plt.plot(df_visual['index'], df_visual['rul_error_mean'])
plt.plot(df_visual['index'], df_visual['rul_error_lower'], linestyle='--')

plt.legend(['RUL Upper', 'RUL Mean', 'RUL Lower'])
plt.ylabel('Error (%)')
plt.xlabel('Current time (min)')

