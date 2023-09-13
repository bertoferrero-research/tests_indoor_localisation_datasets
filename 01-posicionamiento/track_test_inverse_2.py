'''
Analiza la desviación entre la estimación del valor rssi en una posición con el valor auténtico
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import autokeras as ak
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import load_real_track_data_inverse
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_pos_y
from lib.filters.montecarlofilter import monte_carlo_filter
from lib.filters.particlefilter import particle_filter

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_window_median'
model = 'dense'
use_pos_z = False
scale_y = True
remove_not_full_rows = True

#Variables globales
track_file = root_dir+'/preprocessed_inputs/'+input_file_name+'.csv'
output_file = script_dir+'/prediction_output/inverse_'+model+'_'+input_file_name+'_2.csv'
model_dir = script_dir+'/models/'+model
scaler_file = model_dir+'/files/scaler_inverse_2.pkl'
model_file = model_dir+'/files/model_inverse_2.h5' #model_inverse.h5
dim_x = 20.660138018121128
dim_y = 17.64103475472807



#Preparamos los datos
input_data, output_data, sensors_mapping = load_real_track_data_inverse(track_file, scaler_file, use_pos_z, scale_y, remove_not_full_rows, separate_mac_and_pos=False)
#Hacemos categorical el sensor_mac y convertimos a onehot
input_data['sensor_mac'] = pd.Categorical(input_data['sensor_mac'])
input_data= pd.get_dummies(input_data, columns=['sensor_mac'])

#Preparamos el formato de entrada
input_data = input_data.to_numpy().astype(np.float32)

#Cargamos el modelo
model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

#Predecimos
predictions = model.predict(input_data)

#print("-Predicciones-")
#print("Real:")
#print(output_data)
#print("Estimado:")
#print(predictions)

#Desescalamos
with open(scaler_file, 'rb') as scalerFile:
  scaler = pickle.load(scalerFile)
  scalerFile.close()
predictions = scaler.inverse_transform(predictions)
output_data = scaler.inverse_transform(output_data)

#Componemos la salida
output_list = []
for index in range(0, len(predictions)):
  listrow = {
    'predicted_rssi': predictions[index][0],
    'real_rssi': output_data[index][0],
  }
  output_list.append(listrow)
output_data = pd.DataFrame(output_list)

#Preparamos cálculos
output_data['deviation'] = (output_data['predicted_rssi'] - output_data['real_rssi']).abs()



#Imprimimos la desviacion máxima minima y media de X e Y
print("- Desviaciones en predicciones -")
print("Desviación máxima: "+str(output_data['deviation'].max()))
print("Desviación mínima: "+str(output_data['deviation'].min()))
print("Desviación media: "+str(output_data['deviation'].mean()))


#Hacemos la salida
output_data.to_csv(output_file, index=False)