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
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))+'/'
root_dir = script_dir+'../../'
sys.path.insert(1, root_dir)
from lib.datasethelper import parseDevices
from lib.trainingcommon import set_random_seed_value

# Configuración
script_name = 'T1_closer_sensors'
track_file_prefix = 'dataset-track_minsensors'
distance_optimize_config_file_name = 'settings_optimize_constants_1.json'
distance_optimize_config_file = script_dir+distance_optimize_config_file_name
config_file = root_dir+'dataset/cnf/tetam.dev'
number_of_nodes = 4
min_number_of_nodes = 3
dim_x = 20.660138018121128
dim_y = 17.64103475472807
random_seed = 42

windowsettingslist = [
  '10',
  '12'
]

# Cargamos la semilla de los generadores aleatorios
set_random_seed_value(random_seed)

# Definimos la función para calcular la distancia desde rssi
def rssi_to_distance_A1A2(rssi, A1, A2):
    return 10 ** ((A1 - (rssi))/(10 * A2))

# Cargamos el fichero de configuración de dispositivos
beacons, dongles = parseDevices(config_file)
with open(distance_optimize_config_file) as f:
    distance_optimize_config = json.load(f)

# Recorremos cada ventana y ejecutamos la trilateración
model_deviation_data = []
for windowsettings_suffix in windowsettingslist:
    print("---- Predicción de la ventana: "+windowsettings_suffix+" ----")

    # Variables globales
    input_file_name = track_file_prefix+'_'+windowsettings_suffix
    track_file = root_dir+'preprocessed_inputs/paper1/'+input_file_name+'.csv'
    output_dir = script_dir+'output/'+script_name+'/'
    output_file = output_dir+input_file_name+'.csv'
    deviation_file = output_dir+input_file_name+'_deviations.csv'
    figure_file = output_dir+input_file_name+'.png'
    model_deviation_file = output_dir+'/'+track_file_prefix+'_deviations.csv'        

    # Cargamos el csv de la trayectoria
    trajectory = pd.read_csv(track_file)
    rssis = trajectory.iloc[:, 4:]
    output_data = []
    histoypositions = []
    for index, rssi_row in rssis.iterrows():
        # Extraemos las filas con valor distinto a 100 manteniendo el indice
        three_best_dongles = rssi_row[rssi_row != 100]
        # Obtenemos los x mejores dongles (los que tienen menor rssi)
        three_best_dongles = three_best_dongles.sort_values(ascending=False).head(number_of_nodes).index.tolist()
        
        if len(three_best_dongles) < min_number_of_nodes:
            print("No hay suficientes nodos")
            continue

        # Localizamos cada dongle en el listado cargado y acumulamos su posición
        positions = []
        for dongleMac in three_best_dongles:
            # Extraemos la posición
            dongle_positions = dongles[dongleMac][0][:2]
            # Extraemos el rssi
            dongle_rssi = rssi_row[dongleMac]
            # Sacamos las constantes A1 y A2
            A1 = distance_optimize_config[dongleMac]['A1']
            A2 = distance_optimize_config[dongleMac]['A2']
            # Generamos el circulo
            dongle_data = Circle(
                dongle_positions[0], dongle_positions[1], rssi_to_distance_A1A2(dongle_rssi, A1, A2))

            positions.append(dongle_data)

        #Acumulamos datos
        histoypositions.append(Trilateration(positions))
        output_data.append({
            'real_x': trajectory.iloc[index, 1],
            'real_y': trajectory.iloc[index, 2],
        })

    #Calculamos la triangulación de todo el histórico
    solve_history(histoypositions)
    for i in range(len(output_data)):
        output_data[i]['predicted_x'] = histoypositions[i].result.center.x
        output_data[i]['predicted_y'] = histoypositions[i].result.center.y

    output_data = pd.DataFrame(output_data)
    # Preparamos cálculos
    output_data['deviation_x'] = (
        output_data['predicted_x'] - output_data['real_x']).abs()
    output_data['deviation_y'] = (
        output_data['predicted_y'] - output_data['real_y']).abs()
    output_data['eclidean_distance'] = np.sqrt(np.power(
        output_data['deviation_x'], 2) + np.power(output_data['deviation_y'], 2))

    # Imprimimos la desviacion máxima minima y media de X e Y
    print("- Desviaciones en predicciones -")
    print("Desviación máxima X: "+str(output_data['deviation_x'].max()))
    print("Desviación mínima X: "+str(output_data['deviation_x'].min()))
    print("Desviación media X: "+str(output_data['deviation_x'].mean()))
    print("Desviación máxima Y: "+str(output_data['deviation_y'].max()))
    print("Desviación mínima Y: "+str(output_data['deviation_y'].min()))
    print("Desviación media Y: "+str(output_data['deviation_y'].mean()))
    print("Distancia euclídea máxima: " +
        str(output_data['eclidean_distance'].max()))
    print("Distancia euclídea mínima: " +
        str(output_data['eclidean_distance'].min()))
    print("Distancia euclídea media: " +
        str(output_data['eclidean_distance'].mean()))


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


    # Mostramos el grafico
    plt.plot([0, 0, dim_x, dim_x, 0], [0, dim_y,  dim_y, 0, 0], 'go-', label='Real', linewidth=1)
    plt.plot(output_data['real_x'].values, output_data['real_y'].values, 'ro-', label='Real', linewidth=1)
    plt.plot(output_data['predicted_x'].values, output_data['predicted_y'].values, 'mo-', label='Calculada', linewidth=1)
    plt.savefig(figure_file)
    plt.close()


    #Guardamos los datos de desviación para el fichero de desviaciones global
    model_deviation_data.append({
        'windowsettings': windowsettings_suffix,
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
    })

#Imprimimos los datos de desviación para el fichero de desviaciones global
model_deviation_data = pd.DataFrame(model_deviation_data)
model_deviation_data.to_csv(model_deviation_file, index=False)