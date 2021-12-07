



# https://github.com/LahiruJayasinghe/RUL-Net

""" CNN + RNN """


import urllib.request
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import sys

MAXLIFE = 120
SCALE = 1
RESCALE = 1
true_rul = []
test_engine_id = 0
training_engine_id = 0
...
# Download the file from `url` and save it locally under `file_name`:
url1 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD001.txt'
url2 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD002.txt'
url3 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD003.txt'
url4 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/train_FD004.txt'
url5 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD001.txt'
url6 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD002.txt'
url7 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD003.txt'
url8 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/test_FD004.txt'
url9 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD001.txt'
url10 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD002.txt'
url11 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD003.txt'
url12 = 'https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData/RUL_FD004.txt'


urllib.request.urlretrieve(url1, 'train_FD001.txt')
urllib.request.urlretrieve(url2, 'train_FD002.txt')
urllib.request.urlretrieve(url3, 'train_FD003.txt')
urllib.request.urlretrieve(url4, 'train_FD004.txt')
urllib.request.urlretrieve(url5, 'test_FD001.txt')
urllib.request.urlretrieve(url6, 'test_FD002.txt')
urllib.request.urlretrieve(url7, 'test_FD003.txt')
urllib.request.urlretrieve(url8, 'test_FD004.txt')
urllib.request.urlretrieve(url9, 'RUL_FD001.txt')
urllib.request.urlretrieve(url10, 'RUL_FD002.txt')
urllib.request.urlretrieve(url11, 'RUL_FD003.txt')
urllib.request.urlretrieve(url12, 'RUL_FD004.txt')
