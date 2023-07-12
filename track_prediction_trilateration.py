from lib.datasethelper import parseDevices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_window'

#Variables globales
track_file = './dataset_processed_csv/'+input_file_name+'.csv'
output_file = './prediction_output/trilateration_'+input_file_name+'.csv'
config_file = './dataset/cnf/tetam.dev'
dim_x = 20.660138018121128
dim_y = 17.64103475472807

#Cargamos el fichero de configuración de dispositivos
beacons, dongles = parseDevices(config_file)

#Definimos la función de trilateración, consultar notas sobre limitaciones al pie
'''
def rssiToDistance(int: RSSI, int txPower) {
   /* 
    * RSSI in dBm
    * txPower is a transmitter parameter that calculated according to its physic layer and antenna in dBm
    * Return value in meter
    *
    * You should calculate "PL0" in calibration stage:
    * PL0 = txPower - RSSI; // When distance is distance0 (distance0 = 1m or more)
    * 
    * SO, RSSI will be calculated by below formula:
    * RSSI = txPower - PL0 - 10 * n * log(distance/distance0) - G(t)
    * G(t) ~= 0 //This parameter is the main challenge in achiving to more accuracy.
    * n = 2 (Path Loss Exponent, in the free space is 2)
    * distance0 = 1 (m)
    * distance = 10 ^ ((txPower - RSSI - PL0 ) / (10 * n))
    *
    * Read more details:
    *   https://en.wikipedia.org/wiki/Log-distance_path_loss_model
    */
   return pow(10, ((double) (txPower - RSSI - PL0)) / (10 * 2));
}'''

def trilateration_min_cuad(nodos):
    # Convertir RSSI a distancia
    #Basado en el método https://medium.com/beingcoders/convert-rssi-value-of-the-ble-bluetooth-low-energy-beacons-to-meters-63259f307283
    distancias = [(10 ** ((-69 -(nodo[1]))/(10 * 2))) for nodo in nodos]  # Convertir RSSI a distancia

    # Número de nodos y dimensiones (x, y)
    num_nodos = len(nodos)
    dimension = nodos[0][0].shape[0]

    # Matrices para el sistema de ecuaciones
    A = np.zeros((num_nodos - 1, dimension))
    b = np.zeros((num_nodos - 1, 1))

    # Calcular las diferencias de distancia
    for i in range(1, num_nodos):
        A[i-1, :] = 2 * (nodos[i][0] - nodos[0][0])
        b[i-1] = np.square(distancias[0]) - np.square(distancias[i]) - \
                  np.dot(nodos[i][0], nodos[i][0]) + np.dot(nodos[0][0], nodos[0][0])

    # Resolver el sistema de ecuaciones
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calcular la posición estimada
    posicion_estimada = nodos[0][0] + x.flatten()

    return posicion_estimada

#Cargamos el csv de la trayectoria
trajectory = pd.read_csv(track_file)
rssis = trajectory.iloc[:,4:]
output_data = []
for index, rssi_row in rssis.iterrows():
  #Obtenemos los tres mejores dongles (con el valor rssi más alto)
  three_best_dongles = rssi_row.sort_values(ascending=False).head(3).index.tolist()
  
  #Localizamos cada dongle en el listado cargado y acumulamos su posición  
  positions = []
  for dongleMac in three_best_dongles:
    dongle_data = []
    #Extraemos la posición
    dongle_data.append(np.array(dongles[dongleMac][0][:2]))
    #Extraemos el rssi
    dongle_rssi = rssi_row[dongleMac]
    #Añadimos la posición a la lista
    dongle_data.append(dongle_rssi)

    positions.append(dongle_data)
  
  #Calculamos la trilateración
  #x, y = trilateration(positions[0][:2], positions[1][:2], positions[2][:2], positions[0][2], positions[1][2], positions[2][2])
  position = trilateration_min_cuad(positions)
  output_data.append({
    'predicted_x': position[0],
    'predicted_y': position[1],
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