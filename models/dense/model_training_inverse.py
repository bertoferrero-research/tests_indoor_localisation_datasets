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
from sklearn.preprocessing import StandardScaler
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../../')
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data_inverse
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_dataframe


#Variables globales
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
training_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_train_window_median.csv'
test_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_test_window_median.csv'
scaler_file = script_dir+'/files/scaler_inverse.pkl'
model_file = script_dir+'/files/model_inverse.h5'

#Hiperparámetros
embedding_size = 12
batch_size = 1500
epochs = 150


#Cargamos los ficheros
X_train, y_train, X_test, y_test, sensors_mapping = load_training_data_inverse(training_file, test_file, scaler_file, False, True, False, True, separate_mac_and_pos=True)

#####Construimos el modelo
#Capas de entrada
input_positions = Input(X_train[0].shape[1], name='input_positions')
input_mac = Input(X_train[1].shape[1], name='input_mac')

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

#Creamos el modelo
model = Model(inputs=[input_positions, input_mac], outputs=output_layer)
model.compile(loss='mse', optimizer='adam', metrics=['mse'] )
#comparacion de optimizadores https://velascoluis.medium.com/optimizadores-en-redes-neuronales-profundas-un-enfoque-pr%C3%A1ctico-819b39a3eb5
#Seguir luchando por bajar el accuracy en regresion no es buena idea https://stats.stackexchange.com/questions/352036/why-is-accuracy-not-a-good-measure-for-regression-models
print(model.summary())

#Entrenamos
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1)

plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
#print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

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