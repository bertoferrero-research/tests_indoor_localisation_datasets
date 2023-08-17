import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
import sys
sys.path.insert(1, script_dir+'/../')
from lib.trajectories import generar_trayectoria
from lib.trajectories import add_noise_to_track

output_file = script_dir+'/training_tracks/track_1.csv'

##Definimos variables
#Generacion de ruta
dim_x = 20.660138018121128
dim_y = 17.64103475472807
margen = 1
duracion = 50#5*60
velocidad_media = 0.317290997542009
velocidad_maxima = 0.660755316404573
velocidad_minima = 0.00305016804813565
velocidad_muestreo = 0.45
origen_x = 18.031#None
origen_y = 8.465#None
direccion_x_inicial = -1#None
direccion_y_inicial = 0#None
#Adición de errores simulados
max_x_deviation = 9.336936
min_x_deviation = 0.014023066
average_x_deviation = 2.0141628
min_y_deviation = 7.2189074
max_y_deviation = 0.024938583
average_y_deviation = 1.6598095





#Generamos la trayectoria
time, x, y = generar_trayectoria(dim_x, dim_y, margen, duracion, velocidad_media, velocidad_maxima, velocidad_minima, velocidad_muestreo, origen_x, origen_y, direccion_x_inicial, direccion_y_inicial)
#Añadimos el ruido
x_with_noise, y_with_noise = add_noise_to_track(x, y, max_x_deviation, min_x_deviation, average_x_deviation, min_y_deviation, max_y_deviation, average_y_deviation)


#Generamos el dataframe y guardamos el csv
df = pd.DataFrame({'time':time, 'x': x, 'y': y, 'x_with_noise': x_with_noise, 'y_with_noise': y_with_noise})
df.to_csv(output_file, index=False)

#Mostramos la trayectoria

plt.plot([0, 0, dim_y, dim_y, 0], [0, dim_x,  dim_x, 0, 0], 'go-', label='Real', linewidth=1)
plt.plot(
    [0+margen, 0+margen, dim_y-margen, dim_y-margen, 0+margen],
    [0+margen, dim_x-margen, dim_x-margen, 0+margen, 0+margen],
     'bo-', label='Real', linewidth=1)
plt.plot(y, x, 'ro-', linewidth=1)
plt.plot(y_with_noise, x_with_noise, 'mo-', linewidth=1)
plt.show()
