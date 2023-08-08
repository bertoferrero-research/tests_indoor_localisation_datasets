import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Embedding, Dense, concatenate, Flatten
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
scaler_file = script_dir+'/files/scaler_embedded.pkl'
model_file = script_dir+'/files/model_embedded.h5'

#Hiperparámetros
embedding_size = 10
batch_size = 1500
epochs = 25


#Cargamos los ficheros
X_train, y_train, X_test, y_test = load_training_data(training_file, test_file, scaler_file, False, True, False, True)

#Sacamos los nombres de los sensores
sensors = X_train.columns.to_list()
sensors_int = range(len(sensors))
sensors_int_train = np.tile(sensors_int, (X_train.shape[0], 1))
sensors_int_test = np.tile(sensors_int, (X_test.shape[0], 1))

#####Construimos el modelo

#Entradas
input_rssi = Input(shape=X_train.shape[1], name='input_rssi')
input_macs = Input(shape=(len(sensors),), name='input_macs')

#Capa de embedding
embedding_size = 8
macs_embedding = Embedding(input_dim=len(sensors), output_dim=embedding_size, name='macs_embedding')(input_macs)

#Concatenamos las entradas
flatten_macs = Flatten()(macs_embedding)
concat = concatenate([input_rssi, flatten_macs])

#Capas densas
dense_layer = Dense(128, activation='relu')(concat)

#Salida
output = Dense(2, activation='linear')(dense_layer)

#Construimos el modelo y entrenamos
model = Model(inputs=[input_rssi, input_macs], outputs=output)
model.compile(loss='mse', optimizer='adam', metrics=['mse'] )
print(model.summary())

#Entrenamos
history = model.fit([X_train,sensors_int_train], y_train, validation_data=([X_test,sensors_int_test], y_test),
                     batch_size=  batch_size,
                     epochs=  epochs, 
                     verbose=1)

plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate([X_test,sensors_int_test], y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))


#Intentamos estimar los puntos de test
X_test_sample = X_train[:500]
y_test_sample = y_train[:500]
prediction = model.predict([X_test_sample, np.tile(sensors_int, (len(X_test_sample), 1))])
y_pred = pd.DataFrame(prediction, columns=['pos_x', 'pos_y'])
#Desescalamos
y_test_sample = descale_dataframe(y_test_sample)
y_pred = descale_dataframe(y_pred)

plt.plot(y_test_sample['pos_y'].values, y_test_sample['pos_x'].values, 'go-', label='Real', linewidth=1)
plt.plot(y_pred['pos_y'].values, y_pred['pos_x'].values, 'ro-', label='Calculada', linewidth=1)
plt.show()


#Guardamos el modelo
if os.path.exists(model_file):
  os.remove(model_file)
model.save(model_file)