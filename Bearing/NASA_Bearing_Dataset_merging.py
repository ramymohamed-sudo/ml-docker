

# https://www.kaggle.com/yasirabd/nasa-bearing-feature-extraction

# https://www.kaggle.com/brjapon/nasa-bearing-dataset-merging

#from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# - 1st Set
dataset_path_1st = '/kaggle/input/1st_test/1st_test'
# - 2nd Set
dataset_path_2nd = '/kaggle/input/2nd_test/2nd_test'
# - 3rd Set
dataset_path_3rd = '../input/3rd_test/4th_test/txt'

""" SELECT WHICH DATASET TO PROCESS """
dataset_path = dataset_path_3rd #1st # 2nd 3rd

for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print(filename)
