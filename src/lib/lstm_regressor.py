import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) == 0:
    print("Please install Tensorflow that supports GPU")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


class LSTMRegressor():
    def __init__(self, configs, hyper=None):
        self.cf = configs
        self.hyper = hyper

    def make_custom_model(self, x_train):
        # Note:
        # Do activation='tanh' require for scaling -1 to 1?
        # Please, test this

        self.model = Sequential()
        self.model.add(LSTM(self.hyper['neurons'], input_shape=(x_train.shape[1], x_train.shape[2]), stateful=False))
        self.model.add(Dense(1))
        self.model.compile(loss=self.cf['Metrics'], optimizer=self.cf['Optimizer'])
        return self.model

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        self.make_custom_model(X_train)

        if self.cf['Verbose'] != 0:
            print(self.model.summary())

        if self.cf['Save_Results']:
            # Save architecture as an image
            tf.keras.utils.plot_model(self.model, to_file='deep_learning_architecture.png', show_shapes=True, show_layer_names=True)

            model_json = self.model.to_json()
            with open("DL_model.json", "w") as json_file:
                json_file.write(model_json)

        self.model.compile(loss=self.cf['Metrics'], optimizer=self.cf['Optimizer'])

        hist = self.model.fit(X_train, y_train, batch_size=self.cf['Batch_Size'], epochs=self.cf['Epochs'], verbose=self.cf['Verbose'], shuffle=True , validation_data=(X_valid, y_valid))
        return self.model

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        return score
