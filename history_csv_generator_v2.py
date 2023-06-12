import numpy as np
import pandas as pd
import glob
import re
import os.path
import math

#Variables globales
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
fingerprint_history_folder = script_dir+'/dataset/hst/set_1/'                                                 #Ruta donde se encuentran los históricos originales
fingerprint_history_train_file = script_dir+'/dataset_processed_csv/fingerprint_history_train_v2.csv'                                  #Salida del csv de entrenamiento
fingerprint_history_test_file = script_dir+'/dataset_processed_csv/fingerprint_history_test_v2.csv'                                    #Salida del csv de tests
time_grouping_timestamp_difference = 0.02                                                   #Al comprobar la diferencia de tiempos, en cuanto haya una diferencia de mas de este valor se cerrará el grupo anterior
test_data_rate = .2                                                                                 #Porcentaje de filas por posicion a volcar en el archivo de test
sensors_list = ['10','11','12','20','21','22','30','31','32','40', '41', '42']                      #Listado de ids de sensores segun su posición
sensors_mac = []                                                                                    #Extraido de los ficheros
regex_file_position = r"(\d+\.\d+_\d+\.\d+_\d+\.\d+)"                                               #regex para extraer la posición del sensor del nombre del fichero
sensors_header = ['timestamp', 'mac_sensor', 'mac_beacon', 'rssi']                                  #cabeceras de los archivos de los sensores
sensors_dtype = {'timestamp': np.float64, 'mac_sensor': str, 'mac_beacon': str, 'rssi': np.int32}   #tipo de datos en los archivos de los sensores

#Función encargada de cerrar el grupo del segmento sobre la lista final
def close_time_group(data_segment, final_list, sensors_mac):
    subdata = {}
    subdata['timestamp'] = data_segment['timestamp'].mean()
    #Añadimos cada mac
    for mac in sensors_mac:
        macrssi = data_segment[(data_segment['mac_sensor']==mac)]
        if(len(macrssi) > 0):
            subdata[mac] = round(macrssi['rssi'].mean())
        else:
            subdata[mac] = -200

    #Volcamos sobre el dataframe final
    final_list.append(subdata)

#Vamos a agrupar las mediciones de todos los sensores por zona de medición, extraemos para ello todos los ficheros del primer sensor, a partir de él leemos el resto
first_sensor_files = glob.glob(fingerprint_history_folder+'sensor'+sensors_list[0]+'*.mbd')
for first_sensor_file in first_sensor_files:
    #Leemos el contenido del fichero
    #print(first_sensor_file)
    data = pd.read_csv(first_sensor_file, sep=',', names=sensors_header, dtype=sensors_dtype)

    #Extraemos la zona del nombre del fichero vemos de cargar el resto de sensores
    zone = re.search(regex_file_position, first_sensor_file).group(1)
    for i in range(1, len(sensors_list)):
        sensor = sensors_list[i]
        sensor_file = fingerprint_history_folder+'sensor'+sensor+'_'+zone+'.mbd'
        #print(sensor_file)
        data = pd.concat([data, pd.read_csv(sensor_file, sep=',', names=sensors_header, dtype=sensors_dtype)])

    data = data.sort_values(by=['timestamp']).reset_index()
    print(data)
    
    if(len(sensors_mac) == 0):
        sensors_mac = data['mac_sensor'].unique()
    #De la zona extraemos las posiciones x, y y z

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
    data_position = pd.DataFrame(final_list)

    #Por cada sensor mac, extraemos las que no tienen valor -200, calculamos la media y reemplazamos los -200 por este valor
    for mac in sensors_mac:
        macrssi = data_position[(data_position[mac]!=-200)]
        if(len(macrssi) > 0):
            macrssi_mean = round(macrssi[mac].mean())
            data_position[mac] = data_position[mac].replace(-200, macrssi_mean)
    
    #Añadimos las posiciones
    pos_x, pos_y, pos_z = zone.split('_')
    data_position.insert(1, 'pos_x', pos_x)
    data_position.insert(2, 'pos_y', pos_y)
    data_position.insert(3, 'pos_z', pos_z)

    #determinamos el punto de corte en base al ratio de test
    train_test_index = math.floor(len(data_position)*test_data_rate)

    #Escribimos en el csv de salida
    data_position[:-train_test_index].to_csv(fingerprint_history_train_file, mode='a', header=not os.path.exists(fingerprint_history_train_file), index=False)
    data_position[-train_test_index:].to_csv(fingerprint_history_test_file , mode='a', header=not os.path.exists(fingerprint_history_test_file), index=False)