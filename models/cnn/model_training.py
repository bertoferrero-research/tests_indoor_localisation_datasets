import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../../')
from lib.trainingcommon import prepare_training_data
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data

#Variables globales
training_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_train.csv'
test_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_test.csv'
scaler_file = script_dir+'/files/scaler.pkl'
scaler_output_file = script_dir+'/files/scaler_output.pkl'
model_file = script_dir+'/files/model.h5'

#Cargamos los ficheros
X_train, y_train, X_test, y_test = load_training_data(training_file, test_file, scaler_file)

#Escalamos los datos de salida
scaler_pos_x = MinMaxScaler()
scaler_pos_x.fit([[0],[20.66]])
y_train['pos_x'] = scaler_pos_x.transform(y_train[['pos_x']]).flatten()
y_test['pos_x'] = scaler_pos_x.transform(y_test[['pos_x']]).flatten()
scaler_pos_y = MinMaxScaler()
scaler_pos_y.fit([[0],[17.64]])
y_train['pos_y'] = scaler_pos_y.transform(y_train[['pos_y']]).flatten()
y_test['pos_y'] = scaler_pos_y.transform(y_test[['pos_y']]).flatten()

print(X_train)
print(y_train)

#Creamos el modelo
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 2, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(MaxPooling1D(1))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='sigmoid'))  # 3 salidas para las coordenadas (x, y, z)

# Compilar el modelo
model.compile(loss='mse', optimizer='RMSProp', metrics=['accuracy','mse','mae'] )

# Resumen del modelo
model.summary()

#Entrenamos
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  1000,
                     epochs=  25, 
                     verbose=1)

#plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

#Intentamos estimar los puntos de test
print('Estimación de puntos de test:')
X_test_sample = X_train[:100]
y_test_sample = y_train[:100]
y_pred = pd.DataFrame(model.predict(X_test_sample), columns=['pos_x', 'pos_y', 'pos_z'])
print(y_pred)
print(y_test_sample)
plt.plot(y_test_sample['pos_y'].values, y_test_sample['pos_x'].values, 'go-', label='Real', linewidth=1)
plt.plot(y_pred['pos_y'].values, y_pred['pos_x'].values, 'ro-', label='Calculada', linewidth=1)
plt.show()

#Guardamos el modelo
if os.path.exists(model_file):
  os.remove(model_file)
model.save(model_file)
