import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle

#Variables globales
input_file_name = 'track_straight_01_all_sensors.mbd_v2'
model = 'cnn'
track_file = './dataset_processed_csv/'+input_file_name+'.csv'
output_file = './prediction_output/'+model+'_'+input_file_name+'.csv'
model_dir = './models/'+model
scaler_file = model_dir+'/files/scaler.pkl'
model_file = model_dir+'/files/model.h5'

#Funciones
#Preparamos los datos para ser introducidos en el modelo
def prepare_data(data):
    #Extraemos cada parte
    y = data.iloc[:, 1:4]
    X = data.iloc[:, 4:]
    #Normalizamos los rssi a valores positivos de 0 a 1
    #X += 100
    #X /= 100
    #Convertimos a float32 para reducir complejidad
    X = X.astype(np.int32)
    y = y.round(4).astype(np.float32)
    #Por cada columna de X añadimos otra indicando si ese nodo ha de tenerse o no en cuenta
    #nodes = X.columns
    #for node in nodes:
    #  X[node+"_on"] = (X[node] > 0).astype(np.int32)

    #Ordenamos alfabéticamente las columnas de X, asegurandonos de que todos los datasets van en el mismo orden
    X = X.reindex(sorted(X.columns), axis=1)
    #Devolvemos
    return X,y

#Cargamos el fichero y el modelo
track_data = pd.read_csv(track_file)
model = tf.keras.models.load_model(model_file)

#Preparamos los datos
input_data, output_data = prepare_data(track_data)
print(input_data)

#Escalamos
with open(scaler_file, 'rb') as scalerFile:
  scaler = pickle.load(scalerFile)
  scalerFile.close()
input_data = scaler.transform(input_data)

#Si el modelo es cnn, tenemos que darle una forma especial
if model == 'cnn':
  input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)

#Predecimos
predictions = model.predict(input_data)
print(predictions)

#Desescalamos
#with open(scaler_output_file, 'rb') as scalerFile:
#  scaler = pickle.load(scalerFile)
#  scalerFile.close()
#predictions = scaler.inverse_transform(predictions)

#Componemos la salida
output_data = output_data.to_numpy()
output_list = []
for index in range(0, len(predictions)):
  listrow = {
    'predicted_x': predictions[index][0],
    'predicted_y': predictions[index][1],
    'predicted_z': predictions[index][2],
    'real_x': output_data[index][0],
    'real_y': output_data[index][1],
    'real_z': output_data[index][2],
  }
  output_list.append(listrow)
output_data = pd.DataFrame(output_list)

#Preparamos cálculos
output_data['deviation_x'] = (output_data['predicted_x'] - output_data['real_x']).abs()
output_data['deviation_y'] = (output_data['predicted_y'] - output_data['real_y']).abs()
output_data['deviation_z'] = (output_data['predicted_z'] - output_data['real_z']).abs()

#Hacemos la salida
output_data.to_csv(output_file, index=False)

#Mostramos el grafico
plt.plot(output_data['real_x'].values, output_data['real_y'].values, 'go-', label='Real', linewidth=1)
plt.plot(output_data['predicted_x'].values, output_data['predicted_y'].values, 'ro-', label='Calculada', linewidth=1)
plt.show()