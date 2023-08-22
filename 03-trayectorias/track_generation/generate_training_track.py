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
duracion = 50*60#50
velocidad_media = 0.3321928477776
velocidad_maxima = 0.424265045
velocidad_minima = 0.112329279
velocidad_muestreo = 3.3196963983
origen_x = 18.031#None
origen_y = 8.465#None
direccion_x_inicial = -1#None
direccion_y_inicial = 0#None





#Generamos la trayectoria
time, x, y = generar_trayectoria(dim_x, dim_y, margen, duracion, velocidad_media, velocidad_maxima, velocidad_minima, velocidad_muestreo, origen_x, origen_y, direccion_x_inicial, direccion_y_inicial)

#Generamos el dataframe y guardamos el csv
df = pd.DataFrame({'timestamp':time, 'pos_x': x, 'pos_y': y, 'pos_z': 0})
df.to_csv(output_file, index=False)

#Mostramos la trayectoria

plt.plot([0, dim_x,  dim_x, 0, 0], [0, 0, dim_y, dim_y, 0], 'go-', label='Real', linewidth=1)
plt.plot(
    [0+margen, dim_x-margen, dim_x-margen, 0+margen, 0+margen],
    [0+margen, 0+margen, dim_y-margen, dim_y-margen, 0+margen],
     'bo-', label='Real', linewidth=1)
plt.plot(x, y, 'ro-', linewidth=1)
plt.show()
