import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
from lib.trainingcommon import load_real_track_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_pos_y

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_v2'
model = 'cnn'
use_pos_z = False
scale_y = True
remove_not_full_rows = True

#Variables globales
track_file = './dataset_processed_csv/'+input_file_name+'.csv'
output_file = './prediction_output/'+model+'_'+input_file_name+'.csv'
model_dir = './models/'+model
scaler_file = model_dir+'/files/scaler.pkl'
model_file = model_dir+'/files/model.h5'



#Preparamos los datos
input_data, output_data = load_real_track_data(track_file, scaler_file, use_pos_z, scale_y, remove_not_full_rows)
print(input_data)

#Si el modelo es cnn, tenemos que darle una forma especial
if model == 'cnn':
  input_data = input_data.values.reshape(input_data.shape[0], input_data.shape[1], 1)

#Cargamos el modelo
model = tf.keras.models.load_model(model_file)

#Predecimos
predictions = model.predict(input_data)

print("-Predicciones-")
print("Real:")
print(output_data)
print("Estimado:")
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
    #'predicted_z': predictions[index][2],
    'real_x': output_data[index][0],
    'real_y': output_data[index][1],
    #'real_z': output_data[index][2],
  }
  output_list.append(listrow)
output_data = pd.DataFrame(output_list)

#Si se ha escalado la salida, desescalamos
if scale_y:
  output_data['predicted_x'] = descale_pos_x(output_data['predicted_x'])
  output_data['predicted_y'] = descale_pos_y(output_data['predicted_y'])
  output_data['real_x'] = descale_pos_x(output_data['real_x'])
  output_data['real_y'] = descale_pos_y(output_data['real_y'])

#Preparamos cálculos
output_data['deviation_x'] = (output_data['predicted_x'] - output_data['real_x']).abs()
output_data['deviation_y'] = (output_data['predicted_y'] - output_data['real_y']).abs()
#output_data['deviation_z'] = (output_data['predicted_z'] - output_data['real_z']).abs()


#Hacemos la salida
output_data.to_csv(output_file, index=False)

#Mostramos el grafico
plt.plot(output_data['real_y'].values, output_data['real_x'].values, 'go-', label='Real', linewidth=1)
plt.plot(output_data['predicted_y'].values, output_data['predicted_x'].values, 'ro-', label='Calculada', linewidth=1)
plt.show()