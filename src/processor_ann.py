#####
#    Last update: Oct 1 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#    ANN model is grabbed from Dr. Kun Wang code
#####
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras import activations, regularizers#####
#    Last update: Sep 28 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#    ANN model is grabbed from Dr. Kun Wang code
#####
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras import activations, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
def build_model(numInput, numOutput, num_history, DropoutRates=[0.0,0.0]):
    model = Sequential()
    model.add(GRU(32,input_shape=(None, numInput),dropout=DropoutRates[0], recurrent_dropout=0.0, return_sequences=True, activation='relu'))
    model.add(GRU(32,dropout=0.0, recurrent_dropout=DropoutRates[1], return_sequences=False, activation='relu'))
    model.add(Dense(32))
    # model.add(Dense(32,kernel_regularizer=regularizers.l1(0.001)))
    model.add(Activation(activations.relu))
    model.add(Dense(numOutput))
    # model.add(Dense(numOutput, kernel_regularizer=regularizers.l1(0.001)))
    model.add(Activation(activations.linear))
    return model

def train_ann(graph_dir):
    # we know everything is already scaled
    x = np.load(graph_dir+'Training_Data_Input.npy')
    y = np.load(graph_dir+'Training_Data_Output.npy')
    # we know it is related to RNN type
    assert len(x.shape) == 3
    assert len(y.shape) == 2
    (num_points_in, rnn_window_size, num_features_in) = x.shape
    (num_points_out, num_features_out) = y.shape
    assert num_points_in == num_points_out
    model = build_model(num_features_in, num_features_out, rnn_window_size)
    model.compile(optimizer='adam', loss=tf.keras.losses.MSE)
    csv_logger = CSVLogger(graph_dir+'trial_training.csv')
    checkpointer = ModelCheckpoint(filepath=graph_dir+'trial_model_with_Rdropout_20Percent.h5', monitor='loss',
                        verbose=0, save_best_only=True, period=10)
    earlystopper = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=20, min_lr=1e-8,verbose = 1)
    model.fit(x, y, batch_size=128, epochs=2000, callbacks=[checkpointer,csv_logger,earlystopper, reduce_lr],
            shuffle=True, verbose=2)
    trainingloss = pd.read_csv(graph_dir+"trial_training.csv", delimiter=',')
    print("ANN Model Training Finished")
    print("Training Loss:")
    print(trainingloss.tail(), '\n')