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
scaler_file = script_dir+'/files/model_3.pkl'
model_file = script_dir+'/files/model_3.h5'
random_seed = 42

#Hiperparámetros
batch_size = 1500
epochs = 300
loss = 'mse' #'mse'
optimizer = 'adam'
cross_val_splits = 10     #Cantidad de divisiones a realizar en el grupo de entrenamiento para la validación cruzada
cross_val_scoring = 'neg_mean_absolute_error' #'neg_mean_squared_error' #Valor para el scoring de la validación cruzada

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(42)

# ---- Construcción del modelo ---- #

#Cargamos los ficheros
X, y, Xmap = load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

###Construimos el modelo

#Entradas
inputSensors = tf.keras.layers.Input(shape=X.shape[1])
InputMap = tf.keras.layers.Input(shape=Xmap.shape[1])

#Capas ocultas para cada entrada
hiddenLayer_sensors = tf.keras.layers.Dense(10, activation='relu')(inputSensors)
hiddenLayer_sensors = tf.keras.layers.Dense(10, activation='relu')(hiddenLayer_sensors)

hiddenLayer_map = tf.keras.layers.Dense(10, activation='relu')(InputMap)
hiddenLayer_map = tf.keras.layers.Dense(10, activation='relu')(hiddenLayer_map)

#Concatenamos las capas
concat = tf.keras.layers.concatenate([hiddenLayer_sensors, hiddenLayer_map])

#Capas ocultas tras la concatenación
hiddenLayer = tf.keras.layers.Dense(10, activation='relu')(concat)
hiddenLayer = tf.keras.layers.Dense(10, activation='relu')(hiddenLayer)
#hiddenLayer = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(hiddenLayer)
#hiddenLayer = tf.keras.layers.Dropout(0.2)(hiddenLayer)

#Salida
output = tf.keras.layers.Dense(y.shape[1], activation='linear')(hiddenLayer)
#output = tf.keras.layers.Dropout(0.2)(x)
model = tf.keras.models.Model(inputs=[inputSensors,InputMap], outputs=output)

model.compile(loss=loss, optimizer=optimizer, metrics=[loss, 'accuracy'] ) #mse y sgd sugeridos por chatgpt, TODO averiguar y entender por qué
#comparacion de optimizadores https://velascoluis.medium.com/optimizadores-en-redes-neuronales-profundas-un-enfoque-pr%C3%A1ctico-819b39a3eb5
#Seguir luchando por bajar el accuracy en regresion no es buena idea https://stats.stackexchange.com/questions/352036/why-is-accuracy-not-a-good-measure-for-regression-models

# --- Evaluación mediante validación cruzada --- #
#kf = KFold(n_splits=cross_val_splits, shuffle=True)
#estimator = KerasRegressor(build_fn=model, optimizer=optimizer, loss=loss, metrics=[loss], epochs=epochs, batch_size=batch_size, verbose=0)
#cross_val_scores = cross_val_score(estimator, X, y, cv=kf, scoring=cross_val_scoring)


#Entrenamos
X_train, X_test, y_train, y_test, Xmap_train, Xmap_test = train_test_split(X, y, Xmap, test_size=0.2)
history = model.fit([X_train, Xmap_train], y_train, validation_data=([X_test, Xmap_test], y_test),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1)

# Evaluamos usando el test set
score = model.evaluate([X_test, Xmap_test], y_test, verbose=0)

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