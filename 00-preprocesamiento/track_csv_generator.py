import numpy as np
import pandas as pd
import glob
import re
import os.path
import math

#Variables globales
fingerprint_track_folder = './dataset/trk/'                                                 #Ruta donde se encuentran los históricos originales
fingerprint_track_file = 'rectangular_with_rotation_all_sensors.mbd'                        #Fichero a extraer
fingerprint_track_output = './files/track_'+fingerprint_track_file+'.csv'                   #Salida del csv de entrenamiento
min_amount_sensors = 3                                                                      #Número mínimo de sensores por grupo, si es menor la fila se descarta
sensors_mac = []                                                                            #Extraido de los ficheros
#cabeceras de los archivos de los sensores
sensors_header = ['timestamp', 'mac_sensor', 'mac_beacon', 'rssi', 'pos_x', 'pos_y', 'pos_z', 'aruco_pos_1', 'aruco_pos_2', 'aruco_pos_3', 'aruco_pos_4', 'aruco_pos_5', 'aruco_pos_6', 'aruco_pos_7', 'aruco_pos_8', 'aruco_pos_9'] 
#tipo de datos en los archivos de los sensores
sensors_dtype = {'timestamp': np.float64, 'mac_sensor': str, 'mac_beacon': str, 'rssi': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64, 'pos_z': np.float64, 'aruco_pos_1': np.float64, 'aruco_pos_2': np.float64, 'aruco_pos_3': np.float64, 'aruco_pos_4': np.float64, 'aruco_pos_5': np.float64, 'aruco_pos_6': np.float64, 'aruco_pos_7': np.float64, 'aruco_pos_8': np.float64, 'aruco_pos_9': np.float64}   


#Cargamos el fichero
data = pd.read_csv(fingerprint_track_folder+fingerprint_track_file, sep=',', names=sensors_header, dtype=sensors_dtype)
#Retenemos el listado de macs
sensors_mac = data['mac_sensor'].unique()
#Calculamos el tiempo relativo con respecto a la primera toma, con ese dato podemos agrupar por "espacio temporal"
data['relative_timestamp'] = (data['timestamp'] - data['timestamp'][0])*10 #Multiplicandolo por 10 conseguimos agrupar con una granularidad de 0.1s
data['relative_timestamp'] = np.floor(data['relative_timestamp'])
data = data.groupby(['relative_timestamp', 'mac_sensor']).mean(numeric_only=True).reset_index()

#Procesamos la salida
timegroups = data['relative_timestamp'].unique()
data_position = pd.DataFrame({'relative_timestamp': timegroups, 'timestamp': np.nan, 'pos_x': np.nan, 'pos_y': np.nan, 'pos_z': np.nan})
#Añadimos una columna por cada sensor
for sensor_mac in sensors_mac:
    data_position[sensor_mac] = -100 #RSSI mínimo, aunque el minimo segun wikipedia es -80 hay en los datos valores inferiores a este

for index, row in data_position.iterrows():
    timegroup = row['relative_timestamp']
    elements = data[(data['relative_timestamp'] == timegroup)]
    if len(elements) < min_amount_sensors:
        data_position.drop(index)
        continue

    #Calculamos la posición media del grupo
    data_position.at[index, 'pos_x'] = elements['pos_x'].mean()
    data_position.at[index, 'pos_y'] = elements['pos_y'].mean()
    data_position.at[index, 'pos_z'] = elements['pos_z'].mean()
    data_position.at[index, 'timestamp'] = elements['timestamp'].mean()
    #Añadimos cada columna
    for elindex, element in elements.iterrows():
        data_position.at[index, element['mac_sensor']] = element['rssi']

#Guardamos en csv
data_position.to_csv(fingerprint_track_output, index=False)
