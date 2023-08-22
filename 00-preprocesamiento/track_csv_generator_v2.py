import numpy as np
import pandas as pd
import glob
import re
import os.path
import math

#Variables globales
script_dir = os.path.dirname(os.path.abspath(__file__))                                                                  #Referencia al directorio actual, por si ejecutamos el python en otro directorio
fingerprint_track_folder = script_dir+'/dataset/trk/'                                                                    #Ruta donde se encuentran los históricos originales
fingerprint_track_file = 'straight_01_all_sensors.mbd'                                                                   #Fichero a extraer
fingerprint_track_output = script_dir+'/dataset_processed_csv/track_'+fingerprint_track_file+'_v2.csv'                   #Salida del csv de entrenamiento
time_grouping_timestamp_difference = 0.02                                                                                #Al comprobar la diferencia de tiempos, en cuanto haya una diferencia de mas de este valor se cerrará el grupo anterior
sensors_mac = []                                                                                                         #Extraido de los ficheros
#cabeceras de los archivos de los sensores
sensors_header = ['timestamp', 'mac_sensor', 'mac_beacon', 'rssi', 'pos_x', 'pos_y', 'pos_z', 'aruco_pos_1', 'aruco_pos_2', 'aruco_pos_3', 'aruco_pos_4', 'aruco_pos_5', 'aruco_pos_6', 'aruco_pos_7', 'aruco_pos_8', 'aruco_pos_9'] 
#tipo de datos en los archivos de los sensores
sensors_dtype = {'timestamp': np.float64, 'mac_sensor': str, 'mac_beacon': str, 'rssi': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64, 'pos_z': np.float64, 'aruco_pos_1': np.float64, 'aruco_pos_2': np.float64, 'aruco_pos_3': np.float64, 'aruco_pos_4': np.float64, 'aruco_pos_5': np.float64, 'aruco_pos_6': np.float64, 'aruco_pos_7': np.float64, 'aruco_pos_8': np.float64, 'aruco_pos_9': np.float64}   

#Función encargada de cerrar el grupo del segmento sobre la lista final
def close_time_group(data_segment, final_list, sensors_mac):
    subdata = {}
    subdata['timestamp'] = data_segment['timestamp'].mean()
    #Calculamos la posición media del grupo
    subdata['pos_x'] = data_segment['pos_x'].mean()
    subdata['pos_y'] = data_segment['pos_y'].mean()
    subdata['pos_z'] = data_segment['pos_z'].mean()
    #Añadimos cada mac
    for mac in sensors_mac:
        macrssi = data_segment[(data_segment['mac_sensor']==mac)]
        if(len(macrssi) > 0):
            subdata[mac] = round(macrssi['rssi'].mean())
        else:
            subdata[mac] = -200

    #Volcamos sobre el dataframe final
    final_list.append(subdata)


#Cargamos el fichero
data = pd.read_csv(fingerprint_track_folder+fingerprint_track_file, sep=',', names=sensors_header, dtype=sensors_dtype)
#Retenemos el listado de macs
sensors_mac = data['mac_sensor'].unique()

#Recorremos las filas para agrupar por espacio temporal
initial_index = 0
final_list = []
for index, row in data.iterrows():
    #Comprobamos si debemos de cerrar el grupo
    timestamp_diff = row['timestamp']-data['timestamp'][initial_index]
    if timestamp_diff > time_grouping_timestamp_difference:
        #Extraemos el segmento
        data_segment = data[initial_index:index]
        #Cerramos el grupo sobre el dataframe final
        close_time_group(data_segment, final_list, sensors_mac)
        initial_index = index

#Procesamos el ultimo grupo
data_segment = data[initial_index:]
close_time_group(data_segment, final_list, sensors_mac)

#Preparamos el dataframe final
output_dataframe = pd.DataFrame(final_list)

#Guardamos en csv
output_dataframe.to_csv(fingerprint_track_output, index=False)
