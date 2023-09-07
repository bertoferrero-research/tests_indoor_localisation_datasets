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
from lib.trainingcommon import load_real_track_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_pos_y
from lib.trainingcommon import posXYlist_to_grid

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_window_median'#'track_1_rssi'#'track_straight_01_all_sensors.mbd_window_median'
synthtetic_track = False#True#False
model = 'clasificacion1'
remove_not_full_rows = True

#Variables globales
track_file = root_dir+'/preprocessed_inputs/'+("synthetic_tracks/" if synthtetic_track is True else "")+input_file_name+'.csv'
output_file = script_dir+'/prediction_output/'+("synthetic_tracks/" if synthtetic_track is True else "")+model+'_'+input_file_name+'.csv'
model_dir = script_dir+'/models/'+model
scaler_file = model_dir+'/files/scaler.pkl'
model_file = model_dir+'/files/model.h5'
dim_x = 20.660138018121128
dim_y = 17.64103475472807
cell_amount_x = 9
cell_amount_y = 9
random_seed = 42

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

#Preparamos los datos
input_data, output_data = load_real_track_data(track_file, scaler_file, False, False, True)
output_data = posXYlist_to_grid(output_data.to_numpy(), cell_amount_x, cell_amount_y)

#Cargamos el modelo
model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

#Predecimos
predictions = model.predict(input_data)
predictions = np.argmax(predictions, axis=-1)

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
  listrow = {
    'real': output_data[index],
    'predicted': predictions[index],
    'deviation': abs(output_data[index] - predictions[index])
  }
  output_list.append(listrow)
output_data = pd.DataFrame(output_list)


#Imprimimos la desviacion máxima minima y media de X e Y
print("- Desviaciones en predicciones -")
print("Desviación máxima: "+str(output_data['deviation'].max()))
print("Desviación media: "+str(output_data['deviation'].mean()))
print("Desviación mínima: "+str(output_data['deviation'].min()))


#Hacemos la salida
output_data.to_csv(output_file, index=False)