import numpy as np
import pandas as pd
import glob
import re
import os.path
import math
from scipy import stats

#Bucle para la automatización de pruebas
loop_values = ['max', 'min', 'mean', 'median', 'tss']
file_prefix = 'FST3-'

#Valores de defecto de configuración de la ventana
def_min_window_size = 0.5                                                                                 #Tamaño mínimo de la ventana de agrupación
def_max_window_size = 2                                                                                   #Tamaño máximo de la ventana de agrupación
def_min_entries_per_sensor = 2                                                                            #Número mínimo de entradas por sensor para que el sensor se considere valido
def_min_valid_sensors = 12                                                                                #Número mínimo de sensores validos para que la ventana se considere valida 
def_invalid_sensor_value = 100                                                                            #Valor que se asigna a los sensores invalidos
def_sensor_filtering_tipe = 'median'                                                                      #Tipo de filtrado a aplicar a los sensores validos. Valores posibles: 'mean', 'median', 'mode', 'max', 'min',' tss'

for testing_value in loop_values:
    #Configuración de la ventana
    min_window_size = def_min_window_size  
    max_window_size = def_max_window_size                     
    min_entries_per_sensor = def_min_entries_per_sensor                 
    min_valid_sensors = def_min_valid_sensors                            
    invalid_sensor_value = def_invalid_sensor_value                           
    sensor_filtering_tipe = def_sensor_filtering_tipe

    #Definimos el valor del bucle
    sensor_filtering_tipe = testing_value

    #Variables globales
    script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
    root_dir = script_dir+'/../'                                                                        #Referencia al directorio raiz del proyecto
    fingerprint_history_folder = root_dir+'/dataset/hst/set_1/'                                                 #Ruta donde se encuentran los históricos originales
    fingerprint_history_file = root_dir+'/preprocessed_inputs/paper1/'+file_prefix+'fingerprint_'+str(testing_value)+'.csv'                                             #Salida del csv de tests
    sensors_list = ['10','11','12','20','21','22','30','31','32','40', '41', '42']                      #Listado de ids de sensores segun su posición
    sensors_mac = []                                                                                    #Extraido de los ficheros
    regex_file_position = r"(\d+\.\d+_\d+\.\d+_\d+\.\d+)"                                               #regex para extraer la posición del sensor del nombre del fichero
    sensors_header = ['timestamp', 'mac_sensor', 'mac_beacon', 'rssi']                                  #cabeceras de los archivos de los sensores
    sensors_dtype = {'timestamp': np.float64, 'mac_sensor': str, 'mac_beacon': str, 'rssi': np.int32}   #tipo de datos en los archivos de los sensores


    #Vamos a agrupar las mediciones de todos los sensores por zona de medición, extraemos para ello todos los ficheros del primer sensor, a partir de él leemos el resto
    first_sensor_files = glob.glob(fingerprint_history_folder+'sensor'+sensors_list[0]+'*.mbd')
    for first_sensor_file in first_sensor_files:
        ##
        ##Extacción de todos los datos de todos los sensores en la misma posición
        ##

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
        
        if(len(sensors_mac) == 0):
            sensors_mac = data['mac_sensor'].unique()
        #De la zona extraemos las posiciones x, y y z
        pos_x, pos_y, pos_z = zone.split('_')

        ##
        ## Proceso de agrupado por ventana de tiempo y filtrado
        ##
        registries_pool = []
        fingerprint_data = []
        for index, row in data.iterrows():
            #ACUMULACIÓN DE REGISTROS
            #Retenemos la hora actual
            current_time = row['timestamp']
            #Añadimos registro al pool
            registries_pool.append(row)
            #Limpiamos caducados
            while len(registries_pool) > 0 and current_time - registries_pool[0]['timestamp'] > max_window_size:
                registries_pool.pop(0)

            #COMPROBACIÓN DE VENTANA VÁLIDA
            #Comprobamos si la ventana cumple con la antiguedad minima
            if not(len(registries_pool) > 0 and current_time - registries_pool[0]['timestamp'] >= min_window_size):
                continue
            #Comprobamos cuantos sensores válidos hay en el pool. Para ello debemos anotar cuantos sensores tienen al menos x entradas (segun configuración)
            valid_sensors = 0
            for sensor_mac in sensors_mac:
                if len(list(filter(lambda x: x['mac_sensor'] == sensor_mac, registries_pool))) >= min_entries_per_sensor:
                    valid_sensors += 1
            #Si no hay suficientes sensores validos, continuamos
            if valid_sensors < min_valid_sensors:
                continue

            #FILTRADO DE REGISTROS
            subdata = {}
            for sensor_mac in sensors_mac:
                #Obtenemos los registros del sensor
                sensor_registries = list(filter(lambda x: x['mac_sensor'] == sensor_mac, registries_pool))
                #Comprobamos si el sensor es valido
                if len(sensor_registries) >= min_entries_per_sensor:
                    #Si es valido, aplicamos el filtro
                    if sensor_filtering_tipe == 'mean':
                        subdata[sensor_mac] = math.floor(np.mean(list(map(lambda x: x['rssi'], sensor_registries))))
                    elif sensor_filtering_tipe == 'median':
                        subdata[sensor_mac] = math.floor(np.median(list(map(lambda x: x['rssi'], sensor_registries))))
                    elif sensor_filtering_tipe == 'mode':
                        subdata[sensor_mac] = math.floor(stats.mode(list(map(lambda x: x['rssi'], sensor_registries)))[0])
                    elif sensor_filtering_tipe == 'max':
                        subdata[sensor_mac] = math.floor(np.max(list(map(lambda x: x['rssi'], sensor_registries))))
                    elif sensor_filtering_tipe == 'min':
                        subdata[sensor_mac] = math.floor(np.min(list(map(lambda x: x['rssi'], sensor_registries))))
                    elif sensor_filtering_tipe == 'tss':
                        subdata[sensor_mac] = np.sum(list(map(lambda x: 10**(x['rssi']/10), sensor_registries)))
                    else:
                        raise Exception('Invalid filtering type')
                else:
                    #Si no es valido, asignamos el valor de invalido
                    subdata[sensor_mac] = invalid_sensor_value

            #Vaciamos el pool
            registries_pool = []

            #Añadimos el registro a la lista de registros, en este caso añadimos también la posición
            subdata['timestamp'] = current_time
            subdata['pos_x'] = pos_x
            subdata['pos_y'] = pos_y
            subdata['pos_z'] = pos_z
            fingerprint_data.append(subdata)

        #Convertimos la lista de registros en un dataframe
        data_position = pd.DataFrame(fingerprint_data)

        #TODO esto esta hecho por hacer archivos con el mismo formato que el resto de generadores, si se decide usar este formato hay que quitarlo
        #Ponemos pos_x, pos_y y pos_z al principio
        data_position = data_position[['timestamp', 'pos_x', 'pos_y', 'pos_z'] + sensors_mac.tolist()]

        #Escribimos en el csv de salida
        data_position.to_csv(fingerprint_history_file, mode='a', header=not os.path.exists(fingerprint_history_file), index=False)

            





