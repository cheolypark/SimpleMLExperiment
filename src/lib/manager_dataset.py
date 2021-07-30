from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import warnings
import multiprocessing

# For readability, ignore many warnings
warnings.filterwarnings("ignore")

# Show cpu capacity
print(multiprocessing.cpu_count())


class ManagerDataset():
    def __init__(self):
        self.cf = {}
        # If Time_Window_Mode is true, data in X columns is converted to new data containing time windows
        # This is for kinds of RNN which can encode data in time windows

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def convert_X_data_with_window(self, dataset_X):
        if len(dataset_X) == 0:
            return None

        dataX = []

        for i in range(len(dataset_X) - self.cf['Look_Back_For_Model'] - 1):
            dataX.append(dataset_X[i:(i + self.cf['Look_Back_For_Model'])].values)

        # reshape data ##################################
        # dataX to be [samples, time steps, features]
        samples = len(dataX)
        windows = dataX[0].shape[0]
        features = dataX[0].shape[1]
        dataX = np.reshape(dataX, (samples, windows, features))

        return dataX

    def convert_y_data_with_window(self, dataset_y):
        dataset_y = dataset_y[(self.cf['Look_Back_For_Model'] + 1):]

        return np.array(dataset_y)

    def get_one_hot_data(self, data, categorical_vars):
        refined_data = pd.get_dummies(data, columns=categorical_vars, prefix=categorical_vars)
        return refined_data

    def remove_unnecessary_variables(self, data, unnecessary_vars):
        refined_data = data.copy()

        for c in unnecessary_vars:
            del refined_data[c]

        return refined_data

    def get_numeric_data(self, data, categorical_vars):
        refined_data = data.copy()
        for c in categorical_vars:
            refined_data[c] = refined_data[c].astype('category')
            refined_data[c] = refined_data[c].cat.codes

        return refined_data

    def get_X_y_data(self, data, target_variable):
        y = data[target_variable]
        X = data.drop(target_variable, axis=1)
        self.X = X
        self.y = y
        self.target_variable = target_variable
        return X, y

    def set_data(self, X, y):
        # normalize data
        X = StandardScaler().fit_transform(X)

        # split into training and test part
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.4, random_state=42)

    def set_train_test_data(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def set_train_test_data_by_rows(self, df, x_columns, target, train_columns, test_columns):
        # df: dataframe
        # x_columns: [list of columns used for x variables]
        # target: a y variable
        # train_columns: {a column for row selection: values for training row selection}
        # e.g.,)     {'province': ['Busan']}
        # test_columns: {a column for row selection: values for test row selection}

        # Find train and test column keys
        train_columns_key = list(train_columns.keys())[0]
        test_columns_key = list(test_columns.keys())[0]

        # Take data which are in the values of the column
        df = df.where(df[train_columns_key].isin(train_columns[train_columns_key] + test_columns[test_columns_key]))

        # Get the train and test data from the original data
        self.df_train = df.where(df[train_columns_key].isin(train_columns[train_columns_key])).dropna(subset=[train_columns_key])
        self.df_test = df.where(df[test_columns_key].isin(test_columns[test_columns_key])).dropna(subset=[test_columns_key])

        # Select data in the selected columns (i.e., X and Y variables)
        cols = x_columns + [target]
        self.df_train = self.df_train.get(cols)
        self.df_train = self.df_train.dropna(subset=cols)
        self.df_test = self.df_test.get(cols)
        self.df_test = self.df_test.dropna(subset=cols)

        self.X_train, self.X_test, self.y_train, self.y_test = self.df_train[x_columns], self.df_test[x_columns], self.df_train[target], self.df_test[target]

        # Create a look-back data for a kind of RNN architecture
        # make data sets with time windows applied
        self.X_win_train = self.convert_X_data_with_window(self.X_train)
        self.X_win_test = self.convert_X_data_with_window(self.X_test)
        self.y_win_train = self.convert_y_data_with_window(self.y_train)
        self.y_win_test = self.convert_y_data_with_window(self.y_test)

        self.train_size = len(self.df_train)
        self.test_size = len(self.df_test)

        self.df_train_test = pd.concat((self.df_train, self.df_test), sort=False)
