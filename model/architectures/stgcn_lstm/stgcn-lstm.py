# !pip install keras
# !pip install tensorflow
# !pip install numpy
# !pip install pandas
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    Input,
    LSTM,
    concatenate,
    ConvLSTM2D,
    Conv2D,
    Lambda,
    Reshape,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.nn import softmax, leaky_relu
from tensorflow import expand_dims, einsum
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

import tensorflow as tf

from gcn_layer import GCNLayer
from shared.data_processing import Graph, Data_Loader

class Sgcn_Lstm:  # Adding mp2vkv2
    def __init__(
        self,
        train_x,
        train_y,
        valid_x,
        valid_y,
        AD,
        AD2,
        lr=0.0001,
        epoach=200,
        batch_size=10,
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.AD = AD
        self.AD2 = AD2
        self.lr = lr
        self.epoach = epoach
        self.batch_size = batch_size
        self.num_joints = 25

    def _conv_layer(self, Input, filters):
        x = Conv2D(filters=filters, kernel_size=(1, 1), strides=1, activation="relu")(
            Input
        )
        x = Dropout(0.25)(x)
        return x

    def _gcn_layer(self, AD, x):
        # gcn = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([AD, x])
        gcn = GCNLayer()([AD, x])
        return gcn

    def sgcn(self, Input):
        x = self._conv_layer(Input, 64)
        gcn_1 = self._gcn_layer(self.AD, x)
        y = self._conv_layer(Input, 64)
        gcn_2 = self._gcn_layer(self.AD2, y)
        concatenated_gcn_1_2 = concatenate([gcn_1, gcn_2], axis=-1)

        x = self._conv_layer(concatenated_gcn_1_2, 128)
        gcn_3 = self._gcn_layer(self.AD, x)
        y = self._conv_layer(concatenated_gcn_1_2, 128)
        gcn_4 = self._gcn_layer(self.AD2, y)
        concatenated_gcn_3_4 = concatenate([gcn_3, gcn_4], axis=-1)

        gcn = tf.keras.layers.Reshape(
            target_shape=(
                -1,
                concatenated_gcn_3_4.shape[2] * concatenated_gcn_3_4.shape[3],
            )
        )(concatenated_gcn_3_4)

        return gcn

    def Lstm(self, x):
        rec = LSTM(80, return_sequences=True)(x)
        rec = Dropout(0.25)(rec)
        rec1 = LSTM(40, return_sequences=True)(rec)
        rec1 = Dropout(0.25)(rec1)
        rec2 = LSTM(40, return_sequences=True)(rec1)
        rec2 = Dropout(0.25)(rec2)
        rec3 = LSTM(80)(rec2)
        rec3 = Dropout(0.25)(rec3)
        return Dense(1, activation="linear")(rec3)

    def build(self):
        seq_input = Input(
            shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None
        )
        sgcn_layer = self.sgcn(seq_input)
        lstm_sgcn_layer = self.Lstm(sgcn_layer)
        self.model = Model(seq_input, lstm_sgcn_layer)
        return self.model

    def train(self):
        t = dt.now()

        model = self.build()
        model.compile(
            loss=tf.keras.losses.Huber(delta=0.1), optimizer=Adam(learning_rate=self.lr)
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=25)
        checkpoint = ModelCheckpoint(
            "models/model_ex5.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
            save_freq="epoch",
        )

        history = self.model.fit(
            self.train_x,
            self.train_y,
            validation_data=(self.valid_x, self.valid_y),
            epochs=self.epoach,
            batch_size=self.batch_size,
            callbacks=[checkpoint, early_stopping],
        )

        print("Training time: %s" % (dt.now() - t))

        self.model.save("models/my_model_trained_exercise.keras")

        return history

    def load_wights(self, file_path):
        self.model.load_wights(file_path)

    def save(self, file_path="models/my_model_trained_exercise.keras"):
        self.model.save(file_path)

    def prediction(self, data):
        return self.model.predict(data)