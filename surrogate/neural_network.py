
import time
import json
from datetime import datetime
import numpy as np

import tensorflow as tf
from keras import backend as K

class NeuralNetwork():
    def __init__(self, uid=None) -> None:
        self.dB_dt_std, self.dD_dt_std = 0, 0

        self.model = None
        if uid:
            self.uid = uid
        else:
            self.uid = datetime.now().strftime('%Y%m%d_%H%M')

        self.name = f"NeuralNet_{self.uid}"
        self.hp = {
            'units': (9, 27, 81, 162, 324, 648, 1296),
            'act_fun': 'relu',
            'learning_rate': 1E-4,
            'batch_size': 2 ** 12,
            'l1_reg': 1e-4,
            'n_epochs': 100
        }
        self.paths = {
            'hist': 'data/history/',
            'model': 'data/model/',
            'data': 'data/sample_set/',
            'pred': 'data/prediction/',
        }

        if uid:
            self.model = self.load_model(self.name)

    def save_model(self, model, name):
        model.save(f"{self.paths['model']}{name}.keras")
        self.model = model
    
    def load_model(self, name):
        self.model = tf.keras.models.load_model(f"{self.paths['model']}{name}.keras",  custom_objects={'custom_mae':self.custom_mae})
    
    def custom_mae(self, y_true, y_pred):
            loss = y_pred - y_true
            loss = loss / [self.dB_dt_std, self.dD_dt_std]
            loss = tf.math.abs(loss)
            loss = tf.experimental.numpy.sum(loss, axis=1) 
            return loss
    
    def train(self, X_train, y_train, X_val, y_val):
        print('Starting Neural Network training...')
    
        # Set a random seed for tensorflow and numpy to ensure reproducibility
        tf.random.set_seed(10)
        np.random.seed(10)
    
        # Obtain the standard deviations of the training data
        self.dB_dt_std = np.std(y_train[:,0])
        self.dD_dt_std = np.std(y_train[:,1])
        
        # Define the model
        nnetwork = tf.keras.Sequential(name=self.name)
        for n_units in self.hp['units']:
            nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=self.hp['act_fun'],
                                                kernel_regularizer=tf.keras.regularizers.l1(self.hp['l1_reg'])))
            nnetwork.add(tf.keras.layers.Dense(2, activation='linear',
                                            kernel_regularizer=tf.keras.regularizers.l1(self.hp['l1_reg'])))
        # Compile and fit the model
        nnetwork.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.hp['learning_rate']), loss=self.custom_mae)
        train_nn_start = time.time()
        history = nnetwork.fit(X_train, y_train, epochs = self.hp['n_epochs'], validation_data = (X_val, y_val),
                                batch_size = self.hp['batch_size'])
        
        # Save history, invoke plot elsewhere.
        with open(f'{self.paths["hist"]}{self.uid}.json', 'w') as f:
            f.write(json.dumps(history.history))

        # Calculate the training time and save the model
        train_nn_end = time.time()
        train_nn_time = (train_nn_end - train_nn_start)/60
        print('NN training time: {:.3g} minutes.'.format(train_nn_time))
    
        self.save_model(nnetwork, self.name)
        print('Successfully completed Neural Network training.')
    
        # Retrieve the loss name
        try:
            loss_name = nnetwork.loss.__name__
        except AttributeError:
            loss_name = nnetwork.loss.name