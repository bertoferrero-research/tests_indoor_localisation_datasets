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
scaler_file = script_dir+'/files/scaler_inverse.pkl'
model_file = script_dir+'/files/model_inverse.h5'
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
X, y, sensors_mapping = load_training_data_inverse(data_file, scaler_file, False, True, True, separate_mac_and_pos=True)

#####Construimos el modelo
#Capas de entrada
input_positions = Input(X[0].shape[1], name='input_positions')
input_mac = Input(X[1].shape[1], name='input_mac')

#Capa de embedding
embedded_mac = Embedding(input_dim=len(sensors_mapping), output_dim=embedding_size, name='embedded_mac')(input_mac)

#Aplanamos la capa de embedding y la concatenamos con las posiciones x e y
flatten_mac = Flatten()(embedded_mac)
concatenated_inputs = concatenate([input_positions, flatten_mac])

#Capas ocultas y de salida
hidden_layer = Dense(128, activation='relu')(concatenated_inputs)
hidden_layer = Dense(128, activation='relu')(hidden_layer)
hidden_layer = Dense(64, activation='relu')(hidden_layer)
hidden_layer = Dense(64, activation='relu')(hidden_layer)
hidden_layer = Dense(32, activation='relu')(hidden_layer)
output_layer = Dense(1, activation='linear')(hidden_layer)

#Mejor diseño hasta ahora
# hidden_layer = Dense(128, activation='relu')(concatenated_inputs)
# hidden_layer = Dense(128, activation='relu')(hidden_layer)
# hidden_layer = Dense(64, activation='relu')(hidden_layer)
# hidden_layer = Dense(64, activation='relu')(hidden_layer)
# hidden_layer = Dense(32, activation='relu')(hidden_layer)
# output_layer = Dense(1, activation='linear')(hidden_layer)

#Creamos el modelo
model = Model(inputs=[input_positions, input_mac], outputs=output_layer)
model.compile(loss=loss, optimizer=optimizer, metrics=[loss] )

#Realizamos evaluación cruzada
#kf = KFold(n_splits=cross_val_splits, shuffle=True)
#cross_val_scores = cross_val_score_multi_input(model, X, y, loss=loss, optimizer=optimizer, metrics=cross_val_scoring, cv=kf, batch_size=batch_size, epochs=epochs, verbose=1)

#Entrenamos
X_pos_train, X_pos_test, X_rssi_train, X_rssi_test, y_train, y_test = train_test_split(X[0].to_numpy() ,X[1].to_numpy(), y.to_numpy(), test_size=0.2)
history = model.fit([X_pos_train, X_rssi_train], y_train, validation_data=([X_pos_test, X_rssi_test], y_test),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1)

# Evaluamos usando el test set
score = model.evaluate([X_pos_test, X_rssi_test], y_test, verbose=0)


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
print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)