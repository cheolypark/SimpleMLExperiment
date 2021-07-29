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
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras import Model


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        # self.dropout1 = Dropout(rate)
        # self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        # attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerRegressor():
    def __init__(self, configs, hyper=None):
        self.cf = configs
        self.hyper = hyper

    def make_custom_model(self, x_train):
        # Only consider the top 20k words
        # vocab_size = 20000
        vocab_size = 1000
        # vocab_size = x_train.shape[1] # 4

        # Only consider the first 200 words of each movie review
        # e.g.,) I like this movie
        #        1 2    8    9
        maxlen = x_train.shape[1] # 4
        # maxlen = x_train.shape[2] # 17

        # Embedding size for each token
        embed_dim = 32

        # Number of attention heads
        num_heads = 1000

        # Hidden layer size in feed forward network inside transformer
        ff_dim = 32

        inputs = Input(shape=(maxlen,))

        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

        x = embedding_layer(inputs)

        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        # x = Dropout(0.1)(x)
        x = Dense(20, activation="relu")(x)
        # x = Dropout(0.1)(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
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
