
import sys
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir+'/../../'  # Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from sklearn.model_selection import train_test_split
from lib.models.trainers import M1Trainer, M2Trainer, M3Trainer, M4Trainer, M5Trainer, M6Trainer, M7Trainer, M8Trainer
import autokeras as ak
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
from lib.trainingcommon import save_model, save_history, set_random_seed_value
from lib.trainingcommon import load_training_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import load_data
import time

# Definimos el listado de modelos a diseñar
models = [
    'M1',
    'M2',
    'M3',
    'M4', 
    'M5', 
    'M6', 
    'M7',
    'M8'
]

# Dataset a emplear
datasets = [
    'minsensors_10',
    'minsensors_12',
]

# Configuración general
executions_per_model = 100
random_seed = 42
models_dir = script_dir+'/models/'
scaler_filename = 'scaler.pkl'
model_filename = 'model.tf'
dim_x = 20.660138018121128
dim_y = 17.64103475472807


# Configramos la semilla aleatoria
set_random_seed_value(random_seed)

#Creamos el acumulador de datos
data = []

# Recorremos cada modelo
for modelName in models:
    #Recorremos cada dataset
    model_deviation_data = []
    model_deviation_file = models_dir+modelName+'/global_deviations.csv'
    for dataset in datasets:
        dataset_path = root_dir+'preprocessed_inputs/paper1/dataset-track_'+dataset+'.csv'

        print('---- Midiendo rendimiento modelo '+modelName + ' - dataset '+dataset+' ----')
        # Definimos rutas
        model_dir = models_dir+modelName+'/'+dataset+'/'
        model_file = model_dir+model_filename
        scaler_file = model_dir+scaler_filename
        output_dir = model_dir+'output_prediction/'
        log_file = model_dir + 'log.txt'  # Ruta del archivo de registro

        # Si no existe el model_file, pasamos
        if not (os.path.isfile(model_file) or os.path.exists(model_file)):
            print('El modelo '+modelName+' con el dataset '+dataset+' no existe')
            continue

        # Aseguramos que exista el directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir+'predictions.csv'
        deviation_file = output_dir+'deviations.csv'
        figure_file = output_dir+'figure.png'

        # Cargamos el trainer
        trainerClass = globals()[modelName+'Trainer']

        # Precargamos modelo y datos
        model = trainerClass.get_model_instance(model_file)

        # Bucles distintos para M3 y M4 uqe para el resto
        if modelName == 'M3' or modelName == 'M4':
            input_data, output_data, input_map_data = trainerClass.get_training_data(dataset_path, scaler_file)
            exec_count = 0
            while exec_count < executions_per_model:
                exec_count += 1
                start = time.time()
                predictions = model.predict([input_data, input_map_data])
                end = time.time()
                elapsed_time = end - start
                data.append({
                    'model': modelName,
                    'dataset': dataset,
                    'execution': exec_count,
                    'elapsed_time_ms': elapsed_time,
                    'input_data_amount': len(input_data),
                    'average_time_per_input': elapsed_time / len(input_data)
                })
        else:
            input_data, output_data = trainerClass.get_training_data(dataset_path, scaler_file)
            exec_count = 0
            while exec_count < executions_per_model:
                exec_count += 1
                start = time.time()
                predictions = model.predict(input_data)
                end = time.time()
                elapsed_time = end - start
                data.append({
                    'model': modelName,
                    'dataset': dataset,
                    'execution': exec_count,
                    'elapsed_time_ms': elapsed_time,
                    'input_data_amount': len(input_data),
                    'average_time_per_input': elapsed_time / len(input_data)
                })

# Guardamos el archivo de salida
df = pd.DataFrame(data)
df.to_csv(models_dir+'predicting_times.csv', index=False)
print('Guardado en '+models_dir+'predicting_times.csv')
print('Proceso finalizado')