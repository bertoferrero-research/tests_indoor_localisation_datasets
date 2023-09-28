import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
import autokeras as ak
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../../'                                                                        #Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import load_data


#Variables globales
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
data_file = root_dir+'preprocessed_inputs/paper1/fingerprint_history_window_3_4_100_median.csv'
track_file = None#root_dir+'preprocessed_inputs/paper1/track_straight_05_all_sensors.mbd_window_3_4_100_median.csv'
scaler_file = script_dir+'/files/model_2_autokeras.pkl'
model_file = script_dir+'/files/model_2_autokeras.tf'
random_seed = 42

#Autokeras config
max_trials = 50
autokeras_project_name = 'dense_model_2'
auokeras_folder = root_dir+'/tmp/autokeras_training/'

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Construcción del modelo ---- #

#Cargamos los ficheros
X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=True, remove_not_full_rows=False)
if track_file is not None:
  track_X, track_y = load_data(track_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=True, remove_not_full_rows=True)
  X = pd.concat([X, track_X])
  y = pd.concat([y, track_y])


#Construimos el modelo
#Nos basamos en el diseño descrito en el paper "Indoor Localization using RSSI and Artificial Neural Network"
inputlength = X.shape[1]
outputlength = y.shape[1]
hiddenLayerLength = round(inputlength*2/3+outputlength, 0)
print("Tamaño de la entrada: "+str(inputlength))
print("Tamaño de la salida: "+str(outputlength))
print("Tamaño de la capa oculta: "+str(hiddenLayerLength))

input = tf.keras.layers.Input(shape=inputlength)

#x = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
hiddenLayer = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
#hiddenLayer = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(hiddenLayer)
#hiddenLayer = tf.keras.layers.Dropout(0.2)(hiddenLayer)

output = tf.keras.layers.Dense(outputlength, activation='linear')(hiddenLayer)
#output = tf.keras.layers.Dropout(0.2)(x)
model = tf.keras.models.Model(inputs=input, outputs=output)

model.compile(loss=loss, optimizer=optimizer, metrics=[loss, 'accuracy'] ) #mse y sgd sugeridos por chatgpt, TODO averiguar y entender por qué
#comparacion de optimizadores https://velascoluis.medium.com/optimizadores-en-redes-neuronales-profundas-un-enfoque-pr%C3%A1ctico-819b39a3eb5
#Seguir luchando por bajar el accuracy en regresion no es buena idea https://stats.stackexchange.com/questions/352036/why-is-accuracy-not-a-good-measure-for-regression-models

# --- Evaluación mediante validación cruzada --- #
#kf = KFold(n_splits=cross_val_splits, shuffle=True)
#estimator = KerasRegressor(build_fn=model, optimizer=optimizer, loss=loss, metrics=[loss], epochs=epochs, batch_size=batch_size, verbose=0)
#cross_val_scores = cross_val_score(estimator, X, y, cv=kf, scoring=cross_val_scoring)


#Entrenamos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

'''
#Intentamos estimar los puntos de test
X_test_sample = X_train#[:5000]
y_test_sample = y_train#[:5000]
prediction = model.predict(X_test_sample)
y_pred = pd.DataFrame(prediction, columns=['pos_x', 'pos_y'])
#Desescalamos
y_test_sample = descale_dataframe(y_test_sample)
y_pred = descale_dataframe(y_pred)

plt.plot(y_test_sample['pos_y'].values, y_test_sample['pos_x'].values, 'go-', label='Real', linewidth=1)
#plt.plot(y_pred['pos_y'].values, y_pred['pos_x'].values, 'ro-', label='Calculada', linewidth=1)
plt.show()
'''

#Guardamos el modelo
if os.path.exists(model_file):
  os.remove(model_file)
model.save(model_file)

#Sacamos valoraciones
print("-- Resumen del modelo:")
print(model.summary())

# print("-- Evaluación cruzada")
# print("Puntuaciones de validación cruzada:", cross_val_scores)
# print("Puntuación media:", cross_val_scores.mean())
# print("Desviación estándar:", cross_val_scores.std())

print("-- Entrenamiento final")
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)