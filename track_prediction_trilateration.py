from lib.datasethelper import parseDevices
import pandas as pd

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

def trilateration(nodo1, nodo2, nodo3, rssi1, rssi2, rssi3):
    # Posiciones de los nodos (x, y)
    x1, y1 = nodo1
    x2, y2 = nodo2
    x3, y3 = nodo3

    # Convertir RSSI a distancia utilizando un modelo de pérdida de trayectoria libre (FSPL)
    distancia1 = 10 ** ((27.55 - rssi1) / 20)
    distancia2 = 10 ** ((27.55 - rssi2) / 20)
    distancia3 = 10 ** ((27.55 - rssi3) / 20)

    # Trilateración
    A = 2 * x2 - 2 * x1
    B = 2 * y2 - 2 * y1
    C = distancia1**2 - distancia2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = 2 * x3 - 2 * x2
    E = 2 * y3 - 2 * y2
    F = distancia2**2 - distancia3**2 - x2**2 + x3**2 - y2**2 + y3**2

    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)

    return x, y

#Cargamos el csv de la trayectoria
trajectory = pd.read_csv(track_file)
rssis = trajectory.iloc[:,4:]
for index, rssi_row in rssis.iterrows():
  #Obtenemos los tres mejores dongles (con el valor rssi más alto)
  three_best_dongles = rssi_row.sort_values(ascending=False).head(3).index.tolist()
  
  #Localizamos cada dongle en el listado cargado y acumulamos su posición  
  positions = []
  for dongleMac in three_best_dongles:
    #Extraemos la posición
    dongle_data = dongles[dongleMac][0][:2]
    #Extraemos el rssi
    dongle_rssi = rssi_row[dongleMac]
    #Añadimos la posición a la lista
    dongle_data.append(dongle_rssi)

    positions.append(dongle_data)
  
  #Calculamos la trilateración
  x, y = trilateration(positions[0][:2], positions[1][:2], positions[2][:2], positions[0][2], positions[1][2], positions[2][2])


  print(x,y)
  break


'''
En este ejemplo, la función trilateration recibe las posiciones de los nodos (nodo1, nodo2, nodo3) y los valores de RSSI (rssi1, rssi2, rssi3). Luego, se convierte el RSSI a distancia utilizando el modelo de pérdida de trayectoria libre (FSPL). Finalmente, se realiza el cálculo de la trilateración para estimar la posición del objeto.

Es importante tener en cuenta que esta aproximación puede ser menos precisa debido a las limitaciones del RSSI como medida de distancia. Para obtener resultados más precisos, es recomendable utilizar tecnologías que proporcionen mediciones de distancia más exactas, como el tiempo de vuelo (ToF) o la intensidad de la señal recibida (RSSI) con trilateración de tiempo de llegada (Time of Arrival, ToA).

'''