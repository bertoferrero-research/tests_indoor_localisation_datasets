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

files = ['fingerprint_history_train', 'fingerprint_history_test', 'track_straight_01_all_sensors.mbd_v2']
for file in files:
    input_data = './dataset_processed_csv/'+file+'.csv'
    output_data = './dataset_processed_csv/'+file+'_inputed.csv'

    #Cargamos el fichero
    df = pd.read_csv(input_data)
    #Separamos las columnas
    sensors_columns = df.iloc[:, 4:]
    other_columns = df.iloc[:, :4]
    #Ordenamos alfabéticamente las columnas asegurandonos de que todos los datasets van en el mismo orden
    sensors_columns = sensors_columns.reindex(sorted(sensors_columns.columns), axis=1)
    #Solicitamos al modelo entrenado que reemplace los -200 por valores predichos
    sensors_columns = imputing_predict_na_data(sensors_columns)

    #Reconstruimos el dataframe juntando las columnas
    df = pd.concat([other_columns, sensors_columns], axis=1)

    #Guardamos el fichero
    df.to_csv(output_data, index=False)
