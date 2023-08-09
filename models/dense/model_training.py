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
from lib.trainingcommon import load_training_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_dataframe


#Variables globales
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
training_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_train_window_median.csv'
test_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_test_window_median.csv'
scaler_file = script_dir+'/files/scaler.pkl'
model_file = script_dir+'/files/model.h5'


#Cargamos los ficheros
X_train, y_train, X_test, y_test = load_training_data(training_file, test_file, scaler_file, False, True, False, True)


print(X_train)
print(y_train)

#Mostramos los valores de la primera columna
#pdTable = pd.DataFrame({'quantity acumulada':X_train.iloc(axis=1)[0]})
#pdTable.plot(kind='box')
#plt.show()

#Construimos el modelo
#Nos basamos en el diseño descrito en el paper "Indoor Localization using RSSI and Artificial Neural Network"
inputlength = X_train.shape[1]
outputlength = y_train.shape[1]
hiddenLayerLength = round(inputlength*2/3+outputlength, 0)
print("Tamaño de la entrada: "+str(inputlength))
print("Tamaño de la salida: "+str(outputlength))
print("Tamaño de la capa oculta: "+str(hiddenLayerLength))

input = tf.keras.layers.Input(shape=inputlength)
#x = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
x = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
x = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(outputlength, activation='linear')(x)
#output = tf.keras.layers.Dropout(0.2)(x)
model = tf.keras.models.Model(inputs=input, outputs=output)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse','mae'] ) #mse y sgd sugeridos por chatgpt, TODO averiguar y entender por qué
#comparacion de optimizadores https://velascoluis.medium.com/optimizadores-en-redes-neuronales-profundas-un-enfoque-pr%C3%A1ctico-819b39a3eb5
#Seguir luchando por bajar el accuracy en regresion no es buena idea https://stats.stackexchange.com/questions/352036/why-is-accuracy-not-a-good-measure-for-regression-models
print(model.summary())

#Entrenamos
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  1500,
                     epochs=  100, 
                     verbose=1)

plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))

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