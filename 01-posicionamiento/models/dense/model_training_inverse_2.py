'''
Este modelo trata de estimar los valores de RSSI partiendo de una posición
'''
from keras.models import Model
from keras.layers import Input, Embedding, Dense, concatenate, Flatten
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../../../')
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data_inverse
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import cross_val_score_multi_input


#Variables globales
data_file = script_dir+'/../../../preprocessed_inputs/fingerprint_history_window_median.csv'
scaler_file = script_dir+'/files/scaler_inverse_2.pkl'
model_file = script_dir+'/files/model_inverse_2.h5'
random_seed = 42

#Hiperparámetros
embedding_size = 12
batch_size = 1500
epochs = 150
loss = 'mse' #'mse'
optimizer = 'adam'
cross_val_splits = 10     #Cantidad de divisiones a realizar en el grupo de entrenamiento para la validación cruzada
cross_val_scoring = 'mse' #'neg_mean_squared_error' #Valor para el scoring de la validación cruzada

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(42)

#Cargamos los ficheros
X, y, sensors_mapping = load_training_data_inverse(data_file, scaler_file, False, True, True, separate_mac_and_pos=False)

#Hacemos categorical el sensor_mac y convertimos a onehot
X['sensor_mac'] = pd.Categorical(X['sensor_mac'])
X = pd.get_dummies(X, columns=['sensor_mac'])

#Preparamos el formato de entrada
X = X.to_numpy().astype(np.float32)


#####Construimos el modelo
#Capas de entrada
input = Input(X.shape[1], name='input')

#Capas ocultas y de salida
hidden_layer = Dense(128, activation='relu')(input)
hidden_layer = Dense(128, activation='relu')(hidden_layer)
hidden_layer = Dense(64, activation='relu')(hidden_layer)
hidden_layer = Dense(64, activation='relu')(hidden_layer)
hidden_layer = Dense(32, activation='relu')(hidden_layer)
output_layer = Dense(1, activation='linear')(hidden_layer)

#Creamos el modelo
model = Model(inputs=input, outputs=output_layer)
model.compile(loss=loss, optimizer=optimizer, metrics=[loss] )

#Realizamos evaluación cruzada
#kf = KFold(n_splits=cross_val_splits, shuffle=True)
#cross_val_scores = cross_val_score_multi_input(model, X, y, loss=loss, optimizer=optimizer, metrics=cross_val_scoring, cv=kf, batch_size=batch_size, epochs=epochs, verbose=1)

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

print("-- Evaluación cruzada")
print("Puntuaciones de validación cruzada:", cross_val_scores)
print("Puntuación media:", cross_val_scores.mean())
print("Desviación estándar:", cross_val_scores.std())

print("-- Entrenamiento final")
print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)