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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../../')
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import descale_dataframe

#Variables globales
training_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_train_window_tss.csv'
test_file = script_dir+'/../../dataset_processed_csv/fingerprint_history_test_window_tss.csv'
scaler_file = script_dir+'/files/scaler.pkl'
scaler_output_file = script_dir+'/files/scaler_output.pkl'
model_file = script_dir+'/files/model.h5'
scale_y = True

#Cargamos los ficheros
print("Cargando datos")
X_train, y_train, X_test, y_test = load_training_data(training_file, test_file, scaler_file, False, scale_y, False, False)
print("Carga de datos limpios")
print(X_train)
print(y_train)

#Creamos el modelo
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 2, activation='relu'))
model.add(MaxPooling1D(2))
#model.add(Conv1D(256, 2, activation='relu'))
#model.add(MaxPooling1D(1))
model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='linear'))  # 3 salidas para las coordenadas (x, y, z)

# Compilar el modelo
model.compile(loss='mse', optimizer='RMSProp', metrics=['accuracy','mse','mae'] )

# Resumen del modelo
model.summary()

#Entrenamos
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  1500,
                     epochs=  75, 
                     verbose=1)

plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

'''
#Intentamos estimar los puntos de test
print('Estimación de puntos de test:')
X_test_sample = X_test[:1000]
y_test_sample = y_test[:1000]
y_pred = pd.DataFrame(model.predict(X_test_sample), columns=['pos_x', 'pos_y'])
#Desescalamos
if scale_y:
  y_test_sample = descale_dataframe(y_test_sample)
  y_pred = descale_dataframe(y_pred)

print(X_test_sample)
print(y_pred)
print(y_test_sample)
plt.plot(y_test_sample['pos_y'].values, y_test_sample['pos_x'].values, 'go-', label='Real', linewidth=1)
plt.plot(y_pred['pos_y'].values, y_pred['pos_x'].values, 'ro-', label='Calculada', linewidth=1)
plt.show()
'''
#Guardamos el modelo
if os.path.exists(model_file):
  os.remove(model_file)
model.save(model_file)
