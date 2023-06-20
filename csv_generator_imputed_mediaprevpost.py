import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, UpSampling1D, Input
from lib.trainingcommon import imputing_predict_na_data

def reemplazar_nan_con_media(df, columna):
    # Crear una copia del DataFrame para evitar modificar el original
    df_reemplazado = df.copy()
    
    # Obtener los índices donde se encuentran los NaN en la columna dada
    indices_nan = df_reemplazado[columna].index[df_reemplazado[columna].isnull()]

    # Obtener los índices del valor válido previo y posterior al NaN
    indices_validos = df_reemplazado[columna].index[df_reemplazado[columna].notnull()]
    
    # Iterar sobre los índices de los NaN
    for indice_nan in indices_nan:
        
        # Buscar el índice del valor válido previo
        indice_previo = indices_validos[indices_validos < indice_nan].max()
        
        # Buscar el índice del valor válido posterior
        indice_posterior = indices_validos[indices_validos > indice_nan].min()
        
        # Comprobar si el NaN es el primer elemento del DataFrame
        if pd.isna(indice_previo):
            indice_previo = indice_posterior
        
        # Comprobar si el NaN es el último elemento del DataFrame
        if pd.isna(indice_posterior):
            indice_posterior = indice_previo
        
        # Calcular la media de los valores previo y posterior
        media = np.mean([df_reemplazado.at[indice_previo, columna], df_reemplazado.at[indice_posterior, columna]])
        
        # Reemplazar el NaN con la media calculada
        df_reemplazado.at[indice_nan, columna] = media
    
    return df_reemplazado

files = ['fingerprint_history_train', 'fingerprint_history_test', 'track_straight_01_all_sensors.mbd_v2']
for file in files:
    input_data = './dataset_processed_csv/'+file+'.csv'
    output_data = './dataset_processed_csv/'+file+'_inputed_mediaprevpost.csv'

    #Cargamos el fichero
    df = pd.read_csv(input_data)
    #Reemplazamos los -200 por NaN
    df = df.replace(-200, np.nan)
    #Extraemos los nombres de los sensores
    sensors = df.columns[4:]
    for sensor in sensors:
        df = reemplazar_nan_con_media(df, sensor)

    #Guardamos el fichero
    df.to_csv(output_data, index=False)



