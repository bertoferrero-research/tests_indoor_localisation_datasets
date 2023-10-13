
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
from easy_trilateration.model import *  
from easy_trilateration.least_squares import easy_least_squares  
from easy_trilateration.graph import *  
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../'
sys.path.insert(1, root_dir)
from lib.datasethelper import parseDevices

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_window'
number_of_nodes = 4

#Variables globales
track_file = root_dir+'dataset_processed_csv/'+input_file_name+'.csv'
output_file = root_dir+'prediction_output/trilateration_m2_'+input_file_name+'.csv'
config_file = root_dir+'dataset/cnf/tetam.dev'
dim_x = 20.660138018121128
dim_y = 17.64103475472807

#Cargamos el fichero de configuración de dispositivos
beacons, dongles = parseDevices(config_file)


#Cargamos el csv de la trayectoria
trajectory = pd.read_csv(track_file)
rssis = trajectory.iloc[:,4:]
output_data = []
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
    #Generamos el circulo
    dongle_data = Circle(dongle_positions[0], dongle_positions[1], (10 ** ((-20 -(dongle_rssi))/(10 * 6))))

    positions.append(dongle_data)
  
  #Calculamos la trilateración
  #x, y = trilateration(positions[0][:2], positions[1][:2], positions[2][:2], positions[0][2], positions[1][2], positions[2][2])
  result, meta = easy_least_squares(positions)

  output_data.append({
    'predicted_x': result.center.x,
    'predicted_y': result.center.y,
    #'predicted_z': predictions[index][2],
    'real_x': trajectory.iloc[index, 1],
    'real_y': trajectory.iloc[index, 2],
    #'real_z': output_data[index][2],
  })

output_data = pd.DataFrame(output_data)
#Preparamos cálculos
output_data['deviation_x'] = (output_data['predicted_x'] - output_data['real_x']).abs()
output_data['deviation_y'] = (output_data['predicted_y'] - output_data['real_y']).abs()
output_data['eclidean_distance'] = np.sqrt(np.power(output_data['deviation_x'], 2) + np.power(output_data['deviation_y'], 2))

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
plt.plot([0, 0, dim_y, dim_y, 0], [0, dim_x,  dim_x, 0, 0], 'go-', label='Real', linewidth=1)
plt.plot(output_data['real_y'].values, output_data['real_x'].values, 'ro-', label='Real', linewidth=1)
plt.plot(output_data['predicted_y'].values, output_data['predicted_x'].values, 'mo-', label='Calculada', linewidth=1)
plt.show()


'''
En este ejemplo, la función trilateration recibe las posiciones de los nodos (nodo1, nodo2, nodo3) y los valores de RSSI (rssi1, rssi2, rssi3). Luego, se convierte el RSSI a distancia utilizando el modelo de pérdida de trayectoria libre (FSPL). Finalmente, se realiza el cálculo de la trilateración para estimar la posición del objeto.

Es importante tener en cuenta que esta aproximación puede ser menos precisa debido a las limitaciones del RSSI como medida de distancia. Para obtener resultados más precisos, es recomendable utilizar tecnologías que proporcionen mediciones de distancia más exactas, como el tiempo de vuelo (ToF) o la intensidad de la señal recibida (RSSI) con trilateración de tiempo de llegada (Time of Arrival, ToA).

'''