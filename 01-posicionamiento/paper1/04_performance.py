
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

# Configuración del entrenamiento
batch_size = 256
max_trials = 50
overwrite = False
tuner = 'bayesian'

# Configuración general
random_seed = 42
models_dir = script_dir+'/models/'
scaler_filename = 'scaler.pkl'
model_filename = 'model.tf'
design_image_filename = 'model_plot.png'
tmp_dirname = root_dir+'/tmp/autokeras_training/'


# Configramos la semilla aleatoria
set_random_seed_value(random_seed)

# Recorremos cada modelo
for modelName in models:
    #Recorremos cada dataset
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
