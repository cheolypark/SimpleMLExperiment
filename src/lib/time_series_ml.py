import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
import time


class TimeSeriesML():
    def __init__(self, configs):
        self.cf = configs
        self.seed = 10
        self.look_back = self.cf['Look_Back']
        self.first_index = {}

    def load_data(self):
        self.excel_data = pd.read_csv(self.cf['Filename'])
        for d in self.cf['Date_Data']:
            self.excel_data[d] = pd.to_datetime(self.excel_data[d])

        if self.cf['Sort'] is not None:
            self.excel_data.sort_values(self.cf['Sort'], inplace=True)
        return self.excel_data

    def split_data(self):
        split = self.cf['Train_Test_Split']

        cols = self.cf['Columns']

        self.dataset = self.excel_data.get(cols)
        # self.dataset.dropna(subset=cols)

        # normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if split is not None:
            scaled_df = self.scaler.fit_transform(self.dataset)
            self.dataset = pd.DataFrame(scaled_df, index=self.dataset.index, columns=self.dataset.columns)

            self.train_size = int(len(self.dataset) * split)
            self.test_size = len(self.dataset) - self.train_size

            self.df_train = self.dataset[0:self.train_size]
            self.df_test = self.dataset[self.train_size:len(self.dataset)]
        else:
            # Data which are in certain values in a column are taken as train and test data
            train_columns = self.cf['Train_Columns']
            test_columns = self.cf['Test_Columns']

            # Find train and test column keys
            train_columns_key = list(train_columns.keys())[0]
            test_columns_key = list(test_columns.keys())[0]

            # Take data which are in the values in the column
            # In other words, from col1: [1, 2, 3, 4], take col1:[1, 4] for training and test.
            # Remaining rows are ignored
            self.dataset = self.excel_data.where(self.excel_data[train_columns_key].isin(train_columns[train_columns_key] + test_columns[test_columns_key]))

            # Select data in the selected columns (i.e., X and Y variables)
            self.dataset = self.dataset.get(cols)
            self.dataset = self.dataset.dropna(subset=cols)

            # Scale the selected data
            scaled_df = self.scaler.fit_transform(self.dataset)

            # Use the scaled data to make the selected data, so that the same index is reserved
            self.dataset = pd.DataFrame(scaled_df, index=self.dataset.index, columns=self.dataset.columns)

            # Get the train and test data from the original data
            df_train = self.excel_data.where(self.excel_data[train_columns_key].isin(train_columns[train_columns_key])).dropna(subset=[train_columns_key])
            df_test = self.excel_data.where(self.excel_data[test_columns_key].isin(test_columns[test_columns_key])).dropna(subset=[test_columns_key])

            # Get the first index of a state and store it as a dict
            states = df_train.groupby(train_columns_key)
            cur_index = 0
            for s, s_df in states:
                self.first_index[cur_index] = s
                cur_index += len(s_df)

            states = df_test.groupby(test_columns_key)
            for s, s_df in states:
                self.first_index[cur_index] = s

            # Get the train and test data from the selected data
            self.df_train = self.dataset.ix[df_train.index].dropna(subset=cols)
            self.df_test = self.dataset.ix[df_test.index].dropna(subset=cols)

            self.train_size = len(self.df_train)
            self.test_size = len(self.df_test)

            self.dataset = pd.concat((self.df_train, self.df_test), sort=False)

        self.target_column_index = self.dataset.columns.get_loc(self.cf['Target'])

        print(f"train_data_size: {self.train_size} test_data_size: {self.test_size}")

        # make data sets with time windows applied
        self.x_train, self.y_train = self.create_window_dataset(self.df_train)
        self.x_test, self.y_test = self.create_window_dataset(self.df_test)

        print(f"x_train_size: {len(self.x_train)} x_test_size: {len(self.x_test)}")

    def create_window_dataset(self, dataset):
        dataX, dataY = [], []

        for i in range(len(dataset) - self.look_back - 1):
            dataX.append(dataset[i:(i + self.look_back)].values)
            dataY.append(dataset.iloc[i + self.look_back, self.target_column_index])

        # reshape data ##################################
        # dataX to be [samples, time steps, features]
        samples = len(dataX)
        windows = dataX[0].shape[0]
        features = dataX[0].shape[1]
        dataX = np.reshape(dataX, (samples, windows, features))

        # dataX to np.array
        dataY = np.array(dataY)

        return dataX, dataY

    def train_lstm(self, x_train=None, y_train=None):
        # create and fit the LSTM network
        ts = time.time()
        self.model = Sequential()
        self.model.add(LSTM(self.cf['Neurons'], input_shape=(x_train.shape[1], x_train.shape[2]), stateful=False))
        self.model.add(Dense(1))
        self.model.compile(loss=self.cf['Metrics'], optimizer=self.cf['Optimizer'])
        self.model.fit(x_train, y_train, nb_epoch=self.cf['Epochs'],
                       batch_size=self.cf['Batch_Size'], verbose=2)

        print(f'LSTM was learned!: {(time.time() - ts)}')

        if 'Learned_File' in self.cf and self.cf['Learned_File'] is not None:
            self.model.save(self.cf['Learned_File'])

    def train_lgbm(self, x_train=None, y_train=None): #df_train_src, df_test_src):
        LGB_PARAMS = {"objective": "regression",
                      "num_leaves": 5,
                      "learning_rate": 0.013,
                      "bagging_fraction": 0.91,
                      "feature_fraction": 0.81,
                      "reg_alpha": 0.13,
                      "reg_lambda": 0.13,
                      "metric": "rmse",
                      "seed": self.seed,
                      'verbose': -1
                      }

        dtrain_cc = lgb.Dataset(x_train, label=y_train)
        # model_cc = lgb.train(LGB_PARAMS, train_set=dtrain_cc, num_boost_round=200)
        self.model = lgb.train(LGB_PARAMS, train_set=dtrain_cc)
        return self.model

    def train(self, x_train=None, y_train=None):
        if x_train is not None:
            self.x_train, self.y_train = x_train, y_train
        else:
            x_train, y_train = self.x_train, self.y_train

        # create and fit the LSTM network
        # self.train_lgbm(x_train, y_train)
        self.train_lstm(x_train, y_train)

    def predict(self, x_test=None, y_test=None):
        if x_test is not None:
            self.x_test, self.y_test = x_test, y_test
        else:
            x_test, y_test = self.x_test, self.y_test

        y_predict = self.model.predict(x_test)

        y_predict = self.inv_transform(y_predict)
        y_test = self.inv_transform(y_test)

        return y_predict, y_test

    def show_data(self):
        # values = dataset.values
        # groups = [0, 1, 2, 3, 5, 6, 7]
        # i = 1
        # pyplot.figure()
        # for group in groups:
        #     pyplot.subplot(len(groups), 1, i)
        #     pyplot.plot(values[:, group])
        #     pyplot.title(dataset.columns[group], y=0.5, loc='right')
        #     i += 1
        # pyplot.show()
        pass

    def show(self, y_predict, y_test):
        # calculate root mean squared error
        if self.cf['Metrics'] == 'RMSE':
            test_score = math.sqrt(mean_squared_error(y_test, y_predict))
        elif self.cf['Metrics'] == 'MAE':
            test_score = mean_absolute_error(y_test, y_predict)

        title = f'Train Length[{len(self.x_train)}]'
        title += f'  Prediction Length[{len(y_predict)}]'
        title += f'  Input Windows[{self.look_back}]'
        title += f'  {self.cf["Metrics"]} Score[%.4f]' % (test_score)

        print(title)

        # shift train predictions for plotting
        actuals = self.inv_transform(self.dataset)
        y_predict_plot = np.empty_like(actuals)
        y_predict_plot[:] = np.nan
        y_predict_plot[len(actuals) - 1 - len(y_predict): len(actuals) - 1] = y_predict

        train_line = np.empty_like(actuals)
        train_line[:] = np.nan
        train_line[self.look_back : len(self.x_train) + self.look_back] = 0

        # plot baseline and prediction
        fig = plt.figure()
        # fig.suptitle('test title')
        ax = fig.gca()
        ax.set_xticks(np.arange(0, len(actuals), 1))
        plt.grid(True)

        # Draw data names
        for k, v in self.first_index.items():
            plt.axvline(k, color='#ABB2B9')
            plt.text(k, 0, v, fontsize=12)

        plt.plot(actuals)
        plt.plot(y_predict_plot)
        plt.plot(train_line)
        plt.xlabel(title)
        plt.show()

    def inv_transform(self, data):
        """
        # - scaler   = the scaler object (it needs an inverse_transform method)
        # - data     = the data to be inverse transformed as a Series, ndarray, ...
        #              (a 1d object you can assign to a df column)
        # - ftName   = the name of the column to which the data belongs
        # - colNames = all column names of the data on which scaler was fit
        #              (necessary because scaler will only accept a df of the same shape as the one it was fit on)
        """
        if isinstance(data, pd.DataFrame):
            temp = data.to_numpy()
        else:
            temp = data

        col_name = self.cf['Target']
        col_names = self.cf['Columns']
        dummy = pd.DataFrame(np.zeros((len(temp), len(col_names))), columns=col_names)
        dummy[col_name] = temp
        dummy = pd.DataFrame(self.scaler.inverse_transform(dummy), columns=col_names)
        return dummy[col_name].values
