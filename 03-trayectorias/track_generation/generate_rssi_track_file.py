import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os.path
import pickle
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../'
import sys
sys.path.insert(1, root_dir)
from lib.trajectories import generar_trayectoria
from lib.trajectories import add_noise_to_track
from lib.datasethelper import parseDevices 
from lib.trainingcommon import scale_pos_x_single, scale_pos_y_single

#Configuración
track_file = 'track_1.csv'
track_output_file = 'track_1_rssi_12h.csv'
track_file = script_dir+'/training_tracks/'+track_file
track_output = root_dir+'preprocessed_inputs/synthetic_tracks/'+track_output_file
config_file = root_dir+'dataset/cnf/tetam.dev'

#Modelos
model_dir = root_dir+'01-posicionamiento/models/dense'
scaler_file = model_dir+'/files/scaler_inverse.pkl'
model_file = model_dir+'/files/model_inverse.h5'

#Si existe el fichero de salida, lo borramos
if os.path.exists(track_output):
    os.remove(track_output)

#Cargamos el listado de sensores
beacons, sensors = parseDevices(config_file)
sensors = [*sensors]
#Ordenamos alfabéticamente
sensors.sort()

#Cargamos el fichero de la trayectoria
df = pd.read_csv(track_file)

#Cargamos el modelo
model = tf.keras.models.load_model(model_file)

#Cargamos el scaler
scalerFile = open(scaler_file, 'rb')
scaler = pickle.load(scalerFile)
scalerFile.close()

#Procesamos cada fila estimando el valor RSSI de cada sensor
output_data = []
buffer_limit = 500
for index, row in df.iterrows():
    row_data = {
        'timestamp': row['timestamp'],
        'pos_x': row['pos_x'],
        'pos_y': row['pos_y'],
        'pos_z': row['pos_z']
    }
    for sensor_i in range(len(sensors)):
        sensor = sensors[sensor_i]
        input = [ pd.DataFrame([[scale_pos_x_single(row['pos_x']), scale_pos_y_single(row['pos_y'])]]), pd.DataFrame([[sensor_i]])]
        rssi = model.predict(input)
        rssi = rssi
        rssi = scaler.inverse_transform(rssi)
        row_data[sensor] = rssi[0][0]
    output_data.append(row_data)
    if len(output_data) >= buffer_limit:
        #Guardamos el fichero
        df = pd.DataFrame(output_data)
        df.to_csv(track_output, index=False, mode='a', header=not os.path.exists(track_output))
        output_data = []
        print('Guardado de datos en fichero intermedio')

#Guardamos el fichero
df = pd.DataFrame(output_data)
df.to_csv(track_output, index=False, mode='a', header=not os.path.exists(track_output))


