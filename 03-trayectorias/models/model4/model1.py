import sys
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir+'/../../../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import cross_val_score_multi_input
from lib.trainingcommon import load_training_data, groupDataForRnn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pickle
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Variables globales
N = 60  # Elementos en la secuencia
track_file = root_dir+'/preprocessed_inputs/synthetic_tracks/track_1_rssi_12h.csv'
scaler_file = script_dir+'/files/scaler.pkl'
model_file = script_dir+'/files/model.h5'
random_seed = 42

# Hiperparámetros
batch_size = 32
epochs = 40
loss = 'mse'  # 'mse'
optimizer = 'adam'
# cross_val_splits = 10     #Cantidad de divisiones a realizar en el grupo de entrenamiento para la validación cruzada
# cross_val_scoring = 'mse' #'neg_mean_squared_error' #Valor para el scoring de la validación cruzada

# Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Preparación de los datos ---- #

# Cargamos los ficheros
X, y = load_training_data(track_file, scaler_file,
                          include_pos_z=False, scale_y=True, remove_not_full_rows=True)
# Realizamos las agrupaciones
X, y = groupDataForRnn(X.to_numpy(), y.to_numpy(), N, fill_empty_heads=True)

# ---- Construcción del modelo ---- #
# Entrada
input_model = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2]))
# Capas intermedias
hidden_layer = tf.keras.layers.LSTM(128, activation='relu')(input_model) #, return_sequences=True
#hidden_layer = tf.keras.layers.LSTM( 64, activation='relu', return_sequences=True)(hidden_layer)
#hidden_layer = tf.keras.layers.LSTM(32, activation='relu')(hidden_layer)
# Salida
output_layer = tf.keras.layers.Dense(
    y.shape[1], activation='linear')(hidden_layer)

# Creamos el modelo
model = tf.keras.models.Model(inputs=input_model, outputs=output_layer)
model.compile(loss=loss, optimizer=optimizer, metrics=[loss])

# Realizamos evaluación cruzada
# kf = KFold(n_splits=cross_val_splits, shuffle=True)
# cross_val_scores = cross_val_score_multi_input(model, X, y, loss=loss, optimizer=optimizer, metrics=cross_val_scoring, cv=kf, batch_size=batch_size, epochs=epochs, verbose=1)

# Entrenamos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    shuffle=False)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

# Guardamos el modelo
if os.path.exists(model_file):
    os.remove(model_file)
model.save(model_file)

# Sacamos valoraciones
print("-- Resumen del modelo:")
print(model.summary())

# print("-- Evaluación cruzada")
# print("Puntuaciones de validación cruzada:", cross_val_scores)
# print("Puntuación media:", cross_val_scores.mean())
# print("Desviación estándar:", cross_val_scores.std())

print("-- Entrenamiento final")
print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)
