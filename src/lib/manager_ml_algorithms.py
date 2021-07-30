from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
# from lightgbm_sklearn import LightGBM
from src.lib.deep_learning_regressor import DeepLearningRegressor
from src.lib.lstm_regressor import LSTMRegressor
from src.lib.transformer_regressor import TransformerRegressor
from tensorflow import keras
# from gpflow_regressor import GPFLOWRegressor
from src.lib.manager_dataset import ManagerDataset
from src.lib.lightgbm_sklearn import LightGBM
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import datetime
import pickle
import os
from sklearn.gaussian_process.kernels import RBF
from src.util.dataframe import get_os_path


class ManagerMLAlgorithms(ManagerDataset):
    def __init__(self):
        super().__init__()
        self.algorithms = []
        self.hyper = {}
        self.ml_algs = ['LR']

    def init_ml_algorithms(self, ml_algs=None, hyper=None):
        self.algorithms.clear()

        if ml_algs is None:
            ml_algs = self.ml_algs
        else:
            self.ml_algs = ml_algs

        if hyper is None:
            hyper = self.hyper
        else:
            self.hyper = hyper

        for alg in ml_algs:
            if alg == 'TRS':
                self.algorithms.append(TransformerRegressor(self.cf, hyper))
            elif alg == 'LSTM':
                self.algorithms.append(LSTMRegressor(self.cf, hyper))
            elif alg == 'DLR':
                self.algorithms.append(DeepLearningRegressor(self.cf, hyper))
            elif alg == 'LGB':
                self.algorithms.append(LightGBM())
            elif alg == 'BRR':
                self.algorithms.append(linear_model.BayesianRidge())
            elif alg == 'RFR':
                self.algorithms.append(RandomForestRegressor(**hyper))
            elif alg == 'DTR':
                self.algorithms.append(DecisionTreeRegressor())
            elif alg == 'GBR':
                self.algorithms.append(GradientBoostingRegressor())
            elif alg == 'LR':
                self.algorithms.append(LinearRegression())
            elif alg == 'GPR':
                gp_kernel = 1.0 * RBF(1.0) + WhiteKernel()
                self.algorithms.append(GaussianProcessRegressor(kernel=gp_kernel, random_state=0))
            elif alg == 'SVR':
                self.algorithms.append(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
            elif alg == 'MLP':
                self.algorithms.append(MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                                                    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                                                    random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                                                    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

    def load_models(self, path):
        files = os.listdir(path)
        print("Found files: ", files)
        for f in files:
            print("Use a file: ", f)

            if self.is_keras_model(f):
                if 'LSTMRegressor' in f:
                    model_class = LSTMRegressor(self.cf)
                elif 'TransformerRegressor' in f:
                    model_class = TransformerRegressor(self.cf)
                elif 'DeepLearningRegressor' in f:
                    model_class = DeepLearningRegressor(self.cf)

                model_class.model = keras.models.load_model(os.path.join(path, f))
            else:
                with open(os.path.join(path, f), 'rb') as fid:
                    ml = pickle.load(fid)
                    self.algorithms.append(ml)

    def train_ML(self, ml_name):
        for clf in self.algorithms:
            name = type(clf).__name__
            if ml_name == name:
                ml = clf.fit(self.X_train, self.y_train)
                return ml

    def prediction_ML(self, file_save=None):
        print(f'#==============================================================#')
        print(f'Start prediction!')
        print(f'#==============================================================#')

        # iterate over classifiers
        results = {}
        for clf in self.algorithms:
            name = type(clf).__name__
            info = ''
            test_X = self.X_test
            test_y = self.y_test

            if name == 'SVC':
                info = clf.kernel
            elif name == 'DeepLearningRegressor':
                info = clf.type
            elif name == 'LSTMRegressor':
                test_X = self.X_win_test
                test_y = self.y_win_test

            predicted = clf.predict(test_X)
            mae = mean_absolute_error(test_y, predicted)

            print(f'{name} {info}\t', mae)
            results[name] = mae

            if file_save is not None:
                df = pd.DataFrame({'Actual': test_y, 'Predicted': predicted.ravel()})
                df.to_csv(f'{file_save}_{name}_{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv')

        return results

    def perform_ML(self, file_save=None):
        if self.cf['Verbose'] != 0:
            print(f'Training size[{len(self.y_train)}]', f'Test size[{len(self.y_test)}]')

        print('Score:')

        # iterate over classifiers
        results = {}
        for alg in self.algorithms:
            name = type(alg).__name__
            train_X = self.X_train
            train_y = self.y_train
            test_X = self.X_test
            test_y = self.y_test

            # Training MLs
            if name == 'LSTMRegressor':
                train_X = self.X_win_train
                train_y = self.y_win_train
                test_X = self.X_win_test
                test_y = self.y_win_test

            if self.is_keras_model(name):
                alg.fit(train_X, train_y, test_X, test_y)
            else:
                alg.fit(train_X, train_y)

            # alg.fit(self.X_train, self.y_train)
            # score = alg.score(self.X_test, self.y_test)

            predicted = alg.predict(test_X)
            mae = mean_absolute_error(test_y, predicted)

            print(f'{name} Score\t', mae)
            results[name] = mae

            # Save the prediction results
            if file_save is not None:
                df = pd.DataFrame({'Actual': test_y, 'Predicted': predicted.ravel()})
                df.to_csv(f'{file_save}_{name}_{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv')

            # Save the learned model
            if self.cf['Learned_Model_Path'] is not None:
                if self.is_keras_model(name):
                    alg.model.save(get_os_path(f"{self.cf['Learned_Model_Path']}{name}"))
                else:
                    # In the multiple experiment case, a last one will overwrite previous file.
                    with open(get_os_path(f"{self.cf['Learned_Model_Path']}{name}.pkl"), 'wb') as fid:
                        pickle.dump(alg, fid)

        return results

    def get_short_name(self, name):
        if name == 'TransformerRegressor':
            return 'TRS'
        elif name == 'LSTMRegressor':
            return 'LSTM'
        elif name == 'SVR':
            return 'SVR'
        elif name == 'DeepLearningRegressor':
            return 'DLR'
        elif name == 'GradientBoostingRegressor':
            return 'GBR'
        elif name == 'DecisionTreeRegressor':
            return 'DTR'
        elif name == 'LinearRegression':
            return 'LR'
        elif name == 'GaussianProcessRegressor':
            return 'GPR'
        elif name == 'MLPRegressor':
            return 'MLP'
        elif name == 'BayesianRidge':
            return 'BRR'
        elif name == 'RandomForestRegressor':
            return 'RFR'
        elif name == 'LightGBM':
            return 'LGB'

    def is_keras_model(self, name):
        if 'LSTMRegressor' in name or 'TransformerRegressor' in name or 'DeepLearningRegressor' in name:
            return True

        return False
