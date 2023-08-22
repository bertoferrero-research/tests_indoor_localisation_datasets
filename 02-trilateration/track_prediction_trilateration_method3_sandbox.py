'''
Este fichero intenta ejecutar trilateración empleando las constantes A1 y A2 optimizadas
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
import json
from easy_trilateration.model import *  
from easy_trilateration.least_squares import *  
from easy_trilateration.graph import *  
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../'
sys.path.insert(1, root_dir)
from lib.datasethelper import parseDevices
from lib.plot.trilateration import plot_step_with_distances

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_window'
distance_optimize_config_file_name =  'settings_optimize_constants_1.json'
number_of_nodes = 3

#Variables globales
track_file                      = root_dir+'dataset_processed_csv/'+input_file_name+'.csv'
output_file                     = root_dir+'prediction_output/trilateration_m2_'+input_file_name+'.csv'
config_file                     = root_dir+'dataset/cnf/tetam.dev'
distance_optimize_config_file   = script_dir+'/output/'+distance_optimize_config_file_name
dim_x = 20.660138018121128
dim_y = 17.64103475472807

#Definimos la función para calcular la distancia desde rssi
def rssi_to_distance_A1A2(rssi, A1, A2):
    return 10 ** ((A1 -(rssi))/(10 * A2))

#Cargamos el fichero de configuración de dispositivos
beacons, dongles = parseDevices(config_file)
with open(distance_optimize_config_file) as f:
    distance_optimize_config = json.load(f)

#Cargamos el csv de la trayectoria
trajectory = pd.read_csv(track_file)
rssis = trajectory.iloc[:,4:]
histoypositions = []
for index, rssi_row in rssis.iterrows():
  #Obtenemos los tres mejores dongles (con el valor rssi más alto)
  three_best_dongles = rssi_row.sort_values(ascending=False).head(number_of_nodes).index.tolist()
  
  #Localizamos cada dongle en el listado cargado y acumulamos su posición  
  positions = []
  for dongleMac in three_best_dongles:
    #Extraemos la posición
    dongle_positions = dongles[dongleMac][0][:2]
    #Extraemos el rssi
    dongle_rssi = rssi_row[dongleMac]
    #Sacamos las constantes A1 y A2
    A1 = distance_optimize_config[dongleMac]['A1']
    A2 = distance_optimize_config[dongleMac]['A2']
    #Generamos el circulo
    dongle_data = Circle(dongle_positions[0], dongle_positions[1], rssi_to_distance_A1A2(dongle_rssi, A1, A2))

    positions.append(dongle_data)

  histoypositions.append(Trilateration(positions))
  #print(three_best_dongles)
  #print(rssi_row)
  #break

hist: [Trilateration] = histoypositions

solve_history(hist)

#a = animate(hist)
plot_step_with_distances(hist[0])

  
