import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def load_history(uid):
    with open(f"data/history/{uid}.json", "r") as f:
        history = json.load(f)
    return history

def load_hp(uid):
    with open(f"data/history/{uid}.json", "r") as f:
        history = json.load(f)
    return history.get("hp")

def plot_history(history):
    plt.figure(figsize=(12,6))
    plt.title(f"Training history\n\nLR: {history['hp']['learning_rate']}, BS: {history['hp']['batch_size']}, L1: {history['hp'].get('l1_reg', 0)} , L2: {history['hp'].get('l2_reg', 0)}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log', base=10) 
    plt.plot(history['loss'], label="Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    val_min = min(history['val_loss'])
    plt.hlines(y=val_min, xmin=0, xmax=len(history['loss'])-1, colors='green', linestyles='--', lw=2, label=f"Min(Val_loss) = {val_min:.3}")
    plt.legend()
    plt.show()

def custom_mae(dB_dt_std, dD_dt_std):
# Define the custom_mae function outside of the class to get around scope errors
    def mae(y_true, y_pred):
        loss = y_pred - y_true
        loss = loss / [dB_dt_std, dD_dt_std]
        loss = tf.math.abs(loss)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss
    return mae

class NeuralNetwork():
    def __init__(self, hp, uid=None) -> None:
        self.dB_dt_std, self.dD_dt_std = 0, 0

        self.model = None
        if uid:
            self.uid = uid
        else:
            self.uid = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.name = f"NeuralNet_{self.uid}"
        self.hp = hp
        self.paths = {
            'hist': 'data/history/',
            'model': 'data/model/',
            'data': 'data/sample_set/',
            'pred': 'data/prediction/',
        }
        self.history = {}

        if uid:
            self.load_model(self.name)

    def save_model(self, model, name):
        if self.model is None:
            raise Exception("No model present to save.")
        model.save(f"{self.paths['model']}{name}.keras")
        self.model = model
    
    def load_model(self, name):
        model = tf.keras.models.load_model(f"{self.paths['model']}{name}.keras", custom_objects={'custom_mae': custom_mae(self.dB_dt_std, self.dD_dt_std)})
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        # Set a random seed for tensorflow and numpy to ensure reproducibility
        tf.random.set_seed(10)
        np.random.seed(10)
    
        # Obtain the standard deviations of the training data
        self.dB_dt_std = np.std(y_train[:, 0])
        self.dD_dt_std = np.std(y_train[:, 1])

        def scheduler(epoch, lr):
            if epoch < self.hp['n_epochs']/2:
                return lr
            else:
                return float(lr * np.exp(-0.025))

        lr_scheduler = LearningRateScheduler(scheduler)
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        nnetwork = tf.keras.Sequential(name=self.name)
        for n_units in self.hp['units']:
            nnetwork.add(tf.keras.layers.Dense(
                units=n_units, 
                activation=self.hp['act_fun'],
                kernel_regularizer=tf.keras.regularizers.l2(self.hp['l2_reg'])
            ))
        nnetwork.add(tf.keras.layers.Dense(
            2, activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(self.hp['l2_reg'])
        ))

        # Compile and fit the model
        nnetwork.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hp['learning_rate']), 
            # loss=tf.keras.losses.MeanAbsoluteError()
            # loss=self.custom_mae
            loss=custom_mae(self.dB_dt_std, self.dD_dt_std)
        )

        train_nn_start = time.time()
        self.history = nnetwork.fit(
            X_train, y_train, 
            epochs=self.hp['n_epochs'], 
            validation_data=(X_val, y_val),
            batch_size=self.hp['batch_size'],
            callbacks=[lr_scheduler, early_stopping],
            **kwargs
        )
        
        # Save history, invoke plot elsewhere.
        self.history.history['hp'] = self.hp
        with open(f'{self.paths["hist"]}{self.uid}.json', 'w') as f:
            f.write(json.dumps(self.history.history))

        # Calculate the training time and save the model
        train_nn_end = time.time()
        train_nn_time = (train_nn_end - train_nn_start) / 60
        # print('NN training time: {:.3g} minutes.'.format(train_nn_time))

        self.model = nnetwork
        # self.save_model(nnetwork, self.name)
    
        # Retrieve the loss name
        try:
            loss_name = nnetwork.loss.__name__
        except AttributeError:
            loss_name = nnetwork.loss.name

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def show_history(self):
        plot_history(load_history(self.uid))