
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ExpSineSquared
from sklearn.gaussian_process.kernels import RBF
rng = np.random.RandomState(0)

#============================================================================
# Get data from csv
province = 'Busan'
# province = 'Daegu'
# target = 'new_case'
target = 'confirmed'
# target = 'pop_new_confirmed'
# target = 'diff_pop_new_confirmed'
# data_path = "../../data/output/gaussian_process_fitting_test.csv"
data_path = "../../data/output/output_06_04_2021-13_30_59.csv"

df = pd.read_csv(data_path)
df = df.loc[df['province'] == province]
df = df[['date', target]]
df = df.dropna()

df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m%d").astype(int)

x_df = df['date'].to_frame().to_numpy()
y_df = df[target].to_numpy()

# New y
y = y_df*100000

# New X
X = np.arange(len(y))*0.1
X = np.reshape(X, (-1, 1))
#=== End =====================================================================


# Fit KernelRidge with parameter selection based on 5-fold cross validation
param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 20)
                         for p in np.logspace(0, 2, 20)]}
kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
stime = time.time()
kr.fit(X, y)
print("Time for KRR fitting: %.3f" % (time.time() - stime))

# gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
# Test for SVM + RBF

gp_kernel = 1.0 * RBF(1.0) + WhiteKernel()
# gp_kernel = DotProduct() + WhiteKernel()
# gpr = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-1, random_state=0)
gpr = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=gp_kernel)
stime = time.time()
gpr.fit(X, y)
print("Time for GPR fitting: %.3f" % (time.time() - stime))

# Predict using kernel ridge
# X_plot = np.linspace(0, 20, 10000)[:, None]
# use from csv data
X_plot = X

stime = time.time()
y_kr = kr.predict(X_plot)
print("Time for KRR prediction: %.3f" % (time.time() - stime))

# Predict using gaussian process regressor
stime = time.time()
y_gpr, y_std = gpr.predict(X_plot, return_std=True)
print("Time for GPR prediction with standard-deviation: %.3f" % (time.time() - stime))

#===================================================================================
# Plot results
plt.figure(figsize=(10, 5))
lw = 2
plt.scatter(X, y, c='k', label='data')
# plt.plot(X_plot, np.sin(X_plot), color='navy', lw=lw, label='True')
plt.plot(X_plot, y_kr, color='turquoise', lw=lw, label='KRR (%s)' % kr.best_params_)
plt.plot(X_plot, y_gpr, color='darkorange', lw=lw, label='GPR (%s)' % gpr.kernel_)
plt.fill_between(X_plot[:, 0], y_gpr - y_std, y_gpr + y_std, color='darkorange', alpha=0.2)
plt.xlabel('data')
plt.ylabel('target')
# plt.xlim(0, 20)
# plt.ylim(-4, 4)
plt.title(f'{province}: GPR versus Kernel Ridge')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()
