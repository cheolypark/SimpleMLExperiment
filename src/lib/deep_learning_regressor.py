from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DeepLearningRegressor():
    def __init__(self, cf, hyper=None):
        self.cf = cf
        self.hyper = hyper

    def nn_block(self, input_layer, size, dropout_rate, activation):
        out_layer = KL.Dense(size, activation=None)(input_layer)
        #out_layer = KL.BatchNormalization()(out_layer)
        out_layer = KL.Activation(activation)(out_layer)
        # out_layer = KL.Dropout(dropout_rate)(out_layer)
        return out_layer

    def nn_block_simple(self, input_layer, size, activation):
        out_layer = KL.Dense(size, activation=activation)(input_layer)
        return out_layer

    def larger_model(self, input_size):
        # create model
        self.model = Sequential()
        self.model.add(Dense(13, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(6, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def wider_model(self, input_size):
        # create model
        self.model = Sequential()
        self.model.add(Dense(20, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def make_custom_model(self, input_size):
        inp = KL.Input(shape=input_size)
        neurons = self.cf['Neurons']

        hidden_layer = self.nn_block_simple(inp, neurons, "relu")
        hidden_layer = self.nn_block_simple(hidden_layer, neurons, "relu")
        hidden_layer = self.nn_block_simple(hidden_layer, neurons, "relu")
        hidden_layer = self.nn_block_simple(hidden_layer, neurons, "relu")
        out = KL.Dense(1, activation="linear")(hidden_layer)

        self.model = tf.keras.models.Model(inputs=[inp], outputs=out)

        return self.model

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        if X_valid is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        input_size = X_train.shape[1]
        if self.cf['DL_TYPE'] == 'custom':
            self.make_custom_model(input_size)
        elif self.cf['DL_TYPE'] == 'larger':
            self.larger_model(input_size)
        elif self.cf['DL_TYPE'] == 'wider':
            self.wider_model(input_size)

        print(self.model.summary())

        # Save architecture as an image
        if self.cf['Save_Results']:
            tf.keras.utils.plot_model(self.model, to_file='deep_learning_architecture.png', show_shapes=True, show_layer_names=True)

            model_json = self.model.to_json()
            with open("DL_model.json", "w") as json_file:
                json_file.write(model_json)

        self.model.compile(loss=self.cf['Metrics'], optimizer=self.cf['Optimizer'])
        # self.model.compile(loss=self.cf['Metrics'], optimizer=Nadam(lr=self.cf['Learning_Rate']))

        hist = self.model.fit(X_train, y_train, batch_size=self.cf['Batch_Size'], epochs=self.cf['Epochs'], verbose=self.cf['Verbose'], shuffle=True , validation_data=(X_valid, y_valid))

        return self.model

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        score = r2_score(y_test, y_pred)
        return score
