
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

# Definimos el listado de modelos a diseñar
models = [
    'M1',
    # 'M2',
    # 'M3',
    # 'M4', 
    # 'M5', 
    # 'M6', 
    # 'M7',
    # 'M8'
]

# Dataset a emplear
datasets = [
    'minsensors_10',
    'minsensors_12',
]

# Configuración general
random_seed = 42
models_dir = script_dir+'/models/'
scaler_filename = 'scaler.pkl'
model_filename = 'model.tf'
dim_x = 20.660138018121128
dim_y = 17.64103475472807


# Configramos la semilla aleatoria
set_random_seed_value(random_seed)

#Creamos el acumulador general
global_data = {}

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

        # Obtenemos la predicción
        predictions, output_data, prediction_metrics = trainerClass.prediction(dataset_path, model_file, scaler_file)

        #Generamos los calculos y ficheros de salida
        output_list = []
        for index in range(0, len(predictions)):
            listrow = {
                'predicted_x': predictions[index][0],
                'predicted_y': predictions[index][1],
                #'predicted_z': predictions[index][2],
                'real_x': output_data[index][0],
                'real_y': output_data[index][1],
                #'real_z': output_data[index][2],
            }
            output_list.append(listrow)
        output_data = pd.DataFrame(output_list)

        #Preparamos cálculos
        output_data['deviation_x'] = (output_data['predicted_x'] - output_data['real_x']).abs()
        output_data['deviation_y'] = (output_data['predicted_y'] - output_data['real_y']).abs()
        output_data['eclidean_distance'] = np.sqrt(np.power(output_data['deviation_x'], 2) + np.power(output_data['deviation_y'], 2))

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

        #Mostramos el grafico de predicción
        plt.plot([0, 0, dim_x, dim_x, 0], [0, dim_y,  dim_y, 0, 0], 'go-', label='Real', linewidth=1)
        plt.plot(output_data['real_x'].values, output_data['real_y'].values, 'ro-', label='Real', linewidth=1)
        plt.plot(output_data['predicted_x'].values, output_data['predicted_y'].values, 'mo-', label='Calculada', linewidth=1)
        plt.savefig(figure_file)
        plt.close()

        #Imprimimos el gráfico

        #Guardamos los datos de desviación para el fichero de desviaciones global
        if not dataset in global_data:
            global_data[dataset] = []
        global_data[dataset].append({
            'model': modelName,
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
        } | prediction_metrics)
        model_deviation_data.append({
            'dataset': dataset,
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
        } | prediction_metrics)

    #Imprimimos los datos de desviación para el fichero de desviaciones global
    model_deviation_data = pd.DataFrame(model_deviation_data)
    model_deviation_data.to_csv(model_deviation_file, index=False)

#Imprimimos un csv por dataset
for dataset in datasets:
    output_file = models_dir+dataset+'_deviations.csv'
    dataset_data = global_data[dataset]
    dataset_data = pd.DataFrame(dataset_data)
    dataset_data.to_csv(output_file, index=False)