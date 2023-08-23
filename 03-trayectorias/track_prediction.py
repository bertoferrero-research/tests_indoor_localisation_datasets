import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
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
N = 3 #Elementos en la secuencia
input_file_name = 'track_straight_01_all_sensors.mbd_window_median'#'track_1_rssi'
synthtetic_track = False
model = 'model1'
use_pos_z = False
scale_y = True
remove_not_full_rows = True
random_seed = 42

#Variables globales
track_file = root_dir+'/preprocessed_inputs/'+("synthetic_tracks/" if synthtetic_track is True else "")+input_file_name+'.csv'
output_file = script_dir+'/prediction_output/'+("synthetic_tracks/" if synthtetic_track is True else "")+model+'_'+input_file_name+'.csv'
model_dir = script_dir+'/models/'+model
scaler_file = model_dir+'/files/scaler.pkl'
model_file = model_dir+'/files/model.h5'
dim_x = 20.660138018121128
dim_y = 17.64103475472807

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)



#Preparamos los datos
input_data, output_data = load_data(track_file, scaler_file, False, use_pos_z, scale_y, remove_not_full_rows)

#Realizamos las agrupaciones
groupedX = []
groupedy = []
for i in range(N, len(input_data)):
    groupedX.append(input_data.iloc[i-N:i])
    groupedy.append(output_data.iloc[i])
input_data = np.array(groupedX)
output_data = np.array(groupedy)

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
output_list = []
for index in range(0, len(predictions)):
  predicted_index = index - N
  if predicted_index < 0:
    continue
  listrow = {
    'predicted_x': predictions[predicted_index][0],
    'predicted_y': predictions[predicted_index][1],
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
print("Desviación máxima Y: "+str(output_data['deviation_y'].max()))
print("Desviación mínima Y: "+str(output_data['deviation_y'].min()))
print("Desviación media Y: "+str(output_data['deviation_y'].mean()))
print("Distancia euclídea máxima: "+str(output_data['eclidean_distance'].max()))
print("Distancia euclídea mínima: "+str(output_data['eclidean_distance'].min()))
print("Distancia euclídea media: "+str(output_data['eclidean_distance'].mean()))


#Hacemos la salida
output_data.to_csv(output_file, index=False)


#Mostramos el grafico
plt.plot([0, 0, dim_x, dim_x, 0], [0, dim_y,  dim_y, 0, 0], 'go-', label='Real', linewidth=1)
plt.plot(output_data['real_x'].values, output_data['real_y'].values, 'ro-', label='Real', linewidth=1)
plt.plot(output_data['predicted_x'].values, output_data['predicted_y'].values, 'mo-', label='Calculada', linewidth=1)
plt.show()