from pandas import read_csv
from numpy import mean
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
import numpy as np

from os import listdir
from os.path import isfile, join

file_path = '../../data/output_prediction_selected'

files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

# model_selected = ['confirmed',
# 				  'predicted_confirmed_LinearRegression',
# 				  'predicted_confirmed_GaussianProcessRegressor',
# 				  'predicted_confirmed_MLPRegressor',
# 				  'predicted_confirmed_SVR',
# 				  'predicted_confirmed_RandomForestRegressor',
# 				  'predicted_confirmed_GradientBoostingRegressor']

# model_selected = ['confirmed',
# 				  'predicted_confirmed_GaussianProcessRegressor',
# 				  'predicted_confirmed_RandomForestRegressor',
# 				  'predicted_confirmed_GradientBoostingRegressor']

model_selected = ['confirmed',
				  'predicted_confirmed_DeepLearningRegressor',
				  'predicted_confirmed_RandomForestRegressor',
				  'predicted_confirmed_LightGBM']

for file in files:
	data = read_csv(file_path + '/' + file, header=0)

	# get a current province
	provinces = data['province'].unique()
	province = provinces[0]

	# get a set of ML model data
	columns = data.columns.tolist()
	models = [f for f in columns if 'confirmed' in f and f in model_selected]
	data = data[models]

	# get ML labels
	models_label = [model.replace('predicted_confirmed_', '') for model in models]

	# plot
	pyplot.title(province)
	pyplot.plot(data)
	# pyplot.plot(predictions, color='red')

	pyplot.legend( models_label)
	# pyplot.show()
	pyplot.savefig(f'../../data/output_figures/{province}.png')
	pyplot.close()

