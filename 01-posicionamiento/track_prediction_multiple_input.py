import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import autokeras as ak
import pickle
import random
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import load_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_pos_y
from lib.filters.montecarlofilter import monte_carlo_filter
from lib.filters.particlefilter import particle_filter

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_window_3_4_100_median'#'track_1_rssi'#'track_straight_01_all_sensors.mbd_window_median'
synthtetic_track = False#True#False
model = 'dense'
use_pos_z = False
scale_y = True

#Variables globales
track_file = root_dir+'/preprocessed_inputs/paper1/'+("synthetic_tracks/" if synthtetic_track is True else "")+input_file_name+'.csv'
output_file = script_dir+'/prediction_output/'+("synthetic_tracks/" if synthtetic_track is True else "")+model+'_'+input_file_name+'.csv'
deviation_file = script_dir+'/prediction_output/'+("synthetic_tracks/" if synthtetic_track is True else "")+model+'_'+input_file_name+'_deviations.csv'
model_dir = script_dir+'/models/'+model
scaler_file = model_dir+'/files/model_3_scaler_autokeras.pkl'
model_file = model_dir+'/files/model_3_autokeras.tf'
dim_x = 20.660138018121128
dim_y = 17.64103475472807



#Preparamos los datos
input_data, output_data, input_map_data = load_data(track_file, scaler_file, include_pos_z=use_pos_z, scale_y=scale_y, not_valid_sensor_value=100, return_valid_sensors_map=True)

#Si el modelo es cnn, tenemos que darle una forma especial
if model == 'cnn':
  input_data = input_data.values.reshape(input_data.shape[0], input_data.shape[1], 1)

#Cargamos el modelo
model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

#Predecimos
predictions = model.predict([input_data, input_map_data])

# print("-Predicciones-")
# print("Real:")
# print(output_data)
# print("Estimado:")
# print(predictions)

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
output_data['eclidean_distance'] = np.sqrt(np.power(output_data['deviation_x'], 2) + np.power(output_data['deviation_y'], 2))
#output_data['deviation_z'] = (output_data['predicted_z'] - output_data['real_z']).abs()



#Imprimimos la desviacion máxima minima y media de X e Y
print("- Desviaciones en predicciones -")
print("Desviación máxima X: "+str(output_data['deviation_x'].max()))
print("Desviación mínima X: "+str(output_data['deviation_x'].min()))
print("Desviación media X: "+str(output_data['deviation_x'].mean()))
print("Desviación X cuartil 25%: "+str(output_data['deviation_x'].quantile(0.25)))
print("Desviación X cuartil 50%: "+str(output_data['deviation_x'].quantile(0.50)))
print("Desviación X cuartil 75%: "+str(output_data['deviation_x'].quantile(0.75)))

print("Desviación máxima Y: "+str(output_data['deviation_y'].max()))
print("Desviación mínima Y: "+str(output_data['deviation_y'].min()))
print("Desviación media Y: "+str(output_data['deviation_y'].mean()))
print("Desviación Y cuartil 25%: "+str(output_data['deviation_y'].quantile(0.25)))
print("Desviación Y cuartil 50%: "+str(output_data['deviation_y'].quantile(0.50)))
print("Desviación Y cuartil 75%: "+str(output_data['deviation_y'].quantile(0.75)))

print("Distancia euclídea máxima: "+str(output_data['eclidean_distance'].max()))
print("Distancia euclídea mínima: "+str(output_data['eclidean_distance'].min()))
print("Distancia euclídea media: "+str(output_data['eclidean_distance'].mean()))
print("Desviación euclídea cuartil 25%: "+str(output_data['eclidean_distance'].quantile(0.25)))
print("Desviación euclídea cuartil 50%: "+str(output_data['eclidean_distance'].quantile(0.50)))
print("Desviación euclídea cuartil 75%: "+str(output_data['eclidean_distance'].quantile(0.75)))

#Guardamos las desviaciones en csv
deviation_values = pd.DataFrame([{
  'min_x': output_data['deviation_x'].min(),
  'max_x': output_data['deviation_x'].max(),
  'mean_x': output_data['deviation_x'].mean(),
  'q25_x': output_data['deviation_x'].quantile(0.25),
  'q50_x': output_data['deviation_x'].quantile(0.50),
  'q75_x': output_data['deviation_x'].quantile(0.75),
  'min_y': output_data['deviation_y'].min(),
  'max_y': output_data['deviation_y'].max(),
  'mean_y': output_data['deviation_y'].mean(),
  'q25_y': output_data['deviation_y'].quantile(0.25),
  'q50_y': output_data['deviation_y'].quantile(0.50),
  'q75_y': output_data['deviation_y'].quantile(0.75),
  'min_euclidean': output_data['eclidean_distance'].min(),
  'max_euclidean': output_data['eclidean_distance'].max(),
  'mean_euclidean': output_data['eclidean_distance'].mean(),
  'q25_euclidean': output_data['eclidean_distance'].quantile(0.25),
  'q50_euclidean': output_data['eclidean_distance'].quantile(0.50),
  'q75_euclidean': output_data['eclidean_distance'].quantile(0.75),
}])

deviation_values.to_csv(deviation_file, index=False)


#Hacemos la salida de todos los datos en bruto
output_data.to_csv(output_file, index=False)


#Mostramos el grafico
plt.plot([0, 0, dim_x, dim_x, 0], [0, dim_y,  dim_y, 0, 0], 'go-', label='Real', linewidth=1)
plt.plot(output_data['real_x'].values, output_data['real_y'].values, 'ro-', label='Real', linewidth=1)
plt.plot(output_data['predicted_x'].values, output_data['predicted_y'].values, 'mo-', label='Calculada', linewidth=1)
plt.show()