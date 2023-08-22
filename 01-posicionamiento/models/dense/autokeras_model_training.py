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
sys.path.insert(1, script_dir+'/../../')
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_dataframe


#Variables globales
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
data_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_window_median.csv'
scaler_file = script_dir+'/files/scaler.pkl'
model_file = script_dir+'/files/autokeras_model.h5'
random_seed = 42

#Hiperparámetros
#batch_size = 1500
#epochs = 100
#loss = 'mae' #'mse'
#optimizer = 'adam'
#cross_val_splits = 10     #Cantidad de divisiones a realizar en el grupo de entrenamiento para la validación cruzada
#cross_val_scoring = 'neg_mean_absolute_error' #'neg_mean_squared_error' #Valor para el scoring de la validación cruzada

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(42)

# ---- Construcción del modelo ---- #

#Cargamos los ficheros
X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=True, remove_not_full_rows=True)


#Construimos el modelo
#Nos basamos en el diseño descrito en el paper "Indoor Localization using RSSI and Artificial Neural Network"
inputlength = X.shape[1]
outputlength = y.shape[1]

input = ak.Input(shape=inputlength)
output = ak.RegressionHead()(input)
model = ak.AutoModel(inputs=input, outputs=output)


#Entrenamos
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),verbose=1) #

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

print("-- Entrenamiento final")
print('Test loss: {:0.4f}'.format(score[0]))

plot_learning_curves(history)