import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
import shutil
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../../../')
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import posXYList_to_dinamic_grid
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import save_model
from lib.trainingcommon import load_data


#Variables globales
data_file = script_dir+'/../../../preprocessed_inputs/fingerprint_history_window_median.csv'
#track_file = script_dir+'/../../../preprocessed_inputs/synthetic_tracks/track_1_rssi_12h.csv'
scaler_file = script_dir+'/files/scaler.pkl'
model_file = script_dir+'/files/model.h5'
random_seed = 42
cell_amount_x = 3
cell_amount_y = 3
zoom_level = 2

#Hiperparámetros
batch_size = 1500
epochs = 1000
loss = 'categorical_crossentropy' #'mse'
optimizer = 'adam'

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Construcción del modelo ---- #

#Cargamos los ficheros
X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
#track_X, track_y = load_data(track_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
#X = pd.concat([X, track_X])
#y = pd.concat([y, track_y])
y = posXYList_to_dinamic_grid(y.to_numpy(), zoom_level, cell_amount_x, cell_amount_y)

#Separamos por dimension
y_dim1 = y[:,0]
y_dim2 = y[:,1]

#Convertimos a categorical
y_dim1 = tf.keras.utils.to_categorical(y_dim1, num_classes=cell_amount_x*cell_amount_y)
y_dim2 = tf.keras.utils.to_categorical(y_dim2, num_classes=cell_amount_x*cell_amount_y)


#Construimos el modelo
inputlength = X.shape[1]
outputlength_dim1 = y_dim1.shape[1]
outputlength_dim2 = y_dim2.shape[1]

#Parte 1 - Dimension 1
input_rssi = tf.keras.layers.Input(shape=X.shape[1], name='input_rssi')
hiddenLayers_d1 = tf.keras.layers.Dense(10, activation='relu', name='hidden_layers_d1_1')(input_rssi)
output_d1 = tf.keras.layers.Dense(outputlength_dim1, activation='softmax', name='output_d1')(hiddenLayers_d1)

#Parte 2 - Dimension 2
concatenate_input_d2 = tf.keras.layers.Concatenate(name='concatenate_input_d2')([input_rssi, output_d1])
hiddenLayers_d2 = tf.keras.layers.Dense(10, activation='relu', name='hidden_layers_d2_1')(concatenate_input_d2)
output_d2 = tf.keras.layers.Dense(outputlength_dim1, activation='softmax', name='output_d2')(hiddenLayers_d2)


model = tf.keras.models.Model(inputs=input_rssi, outputs=[output_d1, output_d2])
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'] ) 
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, restore_best_weights=True)

#Entrenamos
X_train, X_test, y_dim1_train, y_dim1_test, y_dim2_train, y_dim2_test = train_test_split(X, y_dim1, y_dim2, test_size=0.2)
history = model.fit(X_train, [y_dim1_train, y_dim2_train], validation_data=(X_test, [y_dim1_test, y_dim2_test]),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1,
                     callbacks=[callback])

# Evaluamos usando el test set
score = model.evaluate(X_test, [y_dim1_test, y_dim2_test], verbose=0)


#Guardamos el modelo
save_model(model, model_file)

#Sacamos valoraciones
print("-- Resumen del modelo:")
print(model.summary())

print("-- Entrenamiento")
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)