import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../../../')
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import posXYlist_to_grid
from lib.trainingcommon import descale_dataframe


#Variables globales
data_file = script_dir+'/../../../preprocessed_inputs/fingerprint_history_window_median.csv'
scaler_file = script_dir+'/files/scaler.pkl'
model_file = script_dir+'/files/model.h5'
random_seed = 42
cell_amount_x = 9
cell_amount_y = 9

#Hiperparámetros
batch_size = 1500
epochs = 20
loss = 'categorical_crossentropy' #'mse'
optimizer = 'adam'

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Construcción del modelo ---- #

#Cargamos los ficheros
X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)

#Convertimos a categorical
y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)


#Construimos el modelo
#Nos basamos en el diseño descrito en el paper "Indoor Localization using RSSI and Artificial Neural Network"
inputlength = X.shape[1]
outputlength = y.shape[1]

input = tf.keras.layers.Input(shape=inputlength)

#x = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
hiddenLayer = tf.keras.layers.Dense(128, activation='relu')(input)
hiddenLayer = tf.keras.layers.Dense(128, activation='relu')(hiddenLayer)
hiddenLayer = tf.keras.layers.Dense(64, activation='relu')(hiddenLayer)
hiddenLayer = tf.keras.layers.Dense(32, activation='relu')(hiddenLayer)

output = tf.keras.layers.Dense(outputlength, activation='softmax')(hiddenLayer)
model = tf.keras.models.Model(inputs=input, outputs=output)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'] ) 

#Entrenamos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)


#Guardamos el modelo
if os.path.exists(model_file):
  os.remove(model_file)
model.save(model_file)

#Sacamos valoraciones
print("-- Resumen del modelo:")
print(model.summary())

print("-- Entrenamiento")
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)