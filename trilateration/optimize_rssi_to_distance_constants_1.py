'''
Este script pretende optimizar las constantes A1 y A2 de la ecuación de conversión de RSSI a distancia, para que se ajuste mejor a la realidad. 
Para ello, se utiliza el método de mínimos cuadrados, que se basa en minimizar la suma de los cuadrados de los residuos entre los valores estimados y los valores observados, en este caso, la distancia real.
'''

import pandas as pd
import os.path
import sys
import json
import numpy as np
from scipy.optimize import leastsq
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../'
sys.path.insert(1, root_dir)
from lib.datasethelper import parseDevices


#Configuración
input_file_name = 'fingerprint_history_train_window'
#input_file_name = 'track_straight_01_all_sensors.mbd_window'
init_A1 = -35
init_A2 = 3
# Definir los márgenes para C y R
A1_lower_bound = -40
A1_upper_bound = -30
A2_lower_bound = 2
A2_upper_bound = 5

#Variables globales
data_file = root_dir+'dataset_processed_csv/'+input_file_name+'.csv'
output_file = script_dir+'./output/settings_optimize_constants_1.json'
config_file = root_dir+'dataset/cnf/tetam.dev'
initial_guess = [init_A1, init_A2]

'''
Actuaremos por cada fila del fichero de entrenamiento, sacando todos los valores RSSI y distancia por sensor
'''
#Cargamos los ficheros
beacons, sensors = parseDevices(config_file)
training_data = pd.read_csv(data_file)

#Definimos las funciones necesarias
def rssi_to_distance(rssi, A1, A2): #Función de cálculo base
    return 10 ** ((A1 -(rssi))/(10 * A2))

def residuals(params, rssi, distancia_real): #Función que calcula el rendimiento de los parámetros a probar
    A1, A2 = params

    if(A1 < A1_lower_bound or A1 > A1_upper_bound):
        return [9999] * len(rssi)

    if(A2 < A2_lower_bound or A2 > A2_upper_bound):
        return [9999] * len(rssi)

    # Aplicar recorte (clipping)
    # No se puede redondear o se perderá la capacidad de ajuste
    #A1 = np.round(A1)
    #A2 = np.round(A2)
    #A1 = np.float64(max(A1_lower_bound, min(A1_upper_bound, A1)))
    #A2 = np.float64(max(A2_lower_bound, min(A2_upper_bound, A2)))

    return rssi_to_distance(rssi, A1, A2) - distancia_real

#Actuamos por cada sensor
sensor_settings = {}
for sensor_mac, sensor_data in sensors.items():
    #Extraemos una lista con los rssi
    sensor_rssi = training_data[sensor_mac].tolist()
    #Ahora la distancia
    sensor_distance = [
        np.sqrt(
            np.power(abs(row['pos_x'] - sensor_data[0][0]), 2)
             + 
            np.power(abs(row['pos_y'] - sensor_data[0][1]), 2)
            ) 
        for index, row in training_data.iterrows()
        ]

    # Ajuste usando el método de Levenberg-Marquardt
    optimized_params, _ = leastsq(residuals, initial_guess, args=(sensor_rssi, sensor_distance))
    best_A1, best_A2 = optimized_params

    #Retenemos la configuración
    sensor_settings[sensor_mac] = {
        'A1': best_A1,
        'A2': best_A2
    }

#Guardamos el archivo de salida
with open(output_file, "w") as fp:
    json.dump(sensor_settings , fp, indent=4) 

    

