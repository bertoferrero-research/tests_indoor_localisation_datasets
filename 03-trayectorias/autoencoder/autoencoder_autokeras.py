import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
import autokeras as ak
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, UpSampling1D, Input, Conv1DTranspose
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import load_data
from lib.trainingcommon import plot_learning_curves

#Variables globales
track_file = root_dir+'/preprocessed_inputs/fingerprint_history_window_median.csv'
scaler_file = script_dir+'/files/scaler.pkl'
model_file = script_dir+'/files/model.h5'
random_seed = 42

#Hiperparámetros
embedding_size = 12
batch_size = 1250
epochs = 75
loss = 'mse' #'mse'
optimizer = 'RMSProp'

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Preparación de los datos ---- #
X, y = load_data(track_file, scaler_file, train_scaler_file=True)
# Cambiamos la forma de los datos para que sean compatibles con el modelo
X = X.values.reshape(X.shape[0], X.shape[1], 1)
# En este caso X e y son iguales por lo que la y la descartamos
X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.2, random_state=random_seed)


# ---- Construcción del modelo ---- #

#CNN
# Entrada
input_sensors = Input(shape=(X.shape[1], 1)) 
# Encoder...
encoded = Conv1D(64, 3, activation='relu', padding='same')(input_sensors)
encoded = MaxPooling1D(2, padding='same')(encoded)
encoded = Conv1D(128, 2, activation='relu', padding='same')(encoded)
encoded = MaxPooling1D(2, padding='same')(encoded)
encoded = Conv1D(256, 2, activation='relu', padding='same')(encoded)
#encoded = MaxPooling1D(1, padding='same')(encoded)

# Decoder...
decoded = Conv1DTranspose(256, 2, activation='relu', padding='same')(encoded)
#decoded = UpSampling1D(1)(decoded)
decoded = Conv1DTranspose(128, 2, activation='relu', padding='same')(decoded)
decoded = UpSampling1D(2)(decoded)
decoded = Conv1DTranspose(64, 3, activation='relu', padding='same')(decoded)
decoded = UpSampling1D(2)(decoded)


# Capa de salida con 1 convolución
decoded = Conv1D(1, 3, activation='linear', padding='same')(decoded)

# Compilamos y entrenamos
autoencoder = ak.AutoModel(inputs=input_sensors, outputs=decoded, overwrite=True)#Model(inputs=input_sensors, outputs=decoded)
#encoder = Model(inputs=input_sensors, outputs=encoded)
#autoencoder.compile(optimizer='RMSProp', loss='mse', metrics=['mse'])

#history = autoencoder.fit(X_train, y_train, epochs=40, batch_size=1250, validation_data=(X_test, y_test))
history = autoencoder.fit(X_train, y_train, validation_data=(X_test, y_test))

autoencoder = autoencoder.export_model()

# Guardamos el modelo
#encoder.save(model_file)

# ---- Evaluación del modelo ---- #
autoencoder.summary()
#encoder.summary()
score = autoencoder.evaluate(X_test, y_test, verbose=0)
print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))
plot_learning_curves(history)
