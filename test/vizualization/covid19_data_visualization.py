from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.keras.layers as KL
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

import os

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge

import datetime
import gc
from tqdm import tqdm

import xgboost as xgb

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

PATH_TRAIN = "../ml_inputs/TrainMaster.csv"
df_data = pd.read_csv(PATH_TRAIN)

#===============================================================================================#
# Show CC for all states
#===============================================================================================#
g = df_data.groupby("Province_State")["ConfirmedCases"]
row_data = {}
ticks = []
x_ticks_labels = []
confirmed_cases = []
pre_len = 0
for s, v in g:
    row_data[s] = v.tolist()
    length = len(v.tolist())
    confirmed_cases += v.tolist()
    ticks.append(length + pre_len)
    x_ticks_labels.append(f'{s}')
    pre_len += length
    print(s)

# df_over_state = pd.DataFrame(row_data)
# df_over_state.to_csv('df_over_state.csv')

# plot baseline and predictions
fig = plt.figure()
# fig.suptitle('test title')
ax = fig.gca()
ax.set_xticks(ticks)
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=13)
# ax.set_xticks(np.arange(0, len(confirmed_cases), 1))

# plt.grid(True)
plt.plot(confirmed_cases)
# plt.xlabel(title)
plt.show()

