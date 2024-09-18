from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import Huber
from datetime import datetime as dt
import os

class MobileNet:  # Adding mp2vkv2
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


    def build(self):

        return self.model

    def train(self, weights_path):
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

        model_path = os.path.join(weights_path, "model_weights.keras")

        t = dt.now()
        model = self.build()
        model.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=self.lr))

        early_stopping = EarlyStopping(monitor="val_loss", patience=25)

        checkpoint = ModelCheckpoint(
            model_path,
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
        return history

    def load_wights(self, file_path):
        self.model.load_wights(file_path)

    def save(self, file_path):
        self.model.save(file_path)

    def prediction(self, data):
        return self.model.predict(data)