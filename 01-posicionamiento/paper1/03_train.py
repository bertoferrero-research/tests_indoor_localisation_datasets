
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
    #'M1',
    #'M2',
    #'M3',
     #'M4', 
     #'M5', 
     #'M6', 
     #'M7',
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
        dataset_path = root_dir+'preprocessed_inputs/paper1/dataset-fingerprint_'+dataset+'.csv'

        print('---- Entrenando modelo '+modelName + ' - dataset '+dataset+' ----')
        # Definimos rutas
        model_dir = models_dir+modelName+'/'+dataset+'/'
        model_file = model_dir+model_filename
        scaler_file = model_dir+scaler_filename
        model_tmp_dir = tmp_dirname+modelName+'/'+dataset+'/'
        log_file = model_dir + 'log.txt'  # Ruta del archivo de registro

        # Si existe el model_file (el cual es un directorio), pasamos
        if not overwrite and (os.path.isfile(model_file) or os.path.exists(model_file)):
            continue
        
        #Aseguramos la existencia del directorio final
        os.makedirs(model_dir, exist_ok=True)

        # Obtenemos la clase del modelo entrenador
        trainerClass = globals()[modelName+'Trainer']

        # Guardar el log de tensorflow en un archivo
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            open(log_file, "w").close()
        tf.get_logger().addHandler(logging.FileHandler(log_file, 'a'))

        # Mandamos entrenar el modelo
        model, score = trainerClass.train_model(dataset_path=dataset_path, scaler_file=scaler_file, tuner=tuner, tmp_dir=model_tmp_dir,
                                            batch_size=batch_size, designing=False, overwrite=overwrite, max_trials=max_trials, random_seed=random_seed)

        # Guardamos el modelo
        save_model(model, model_file)

        # Imprimimos resultados
        with open(log_file, 'a') as f:
            print("-- Resumen del modelo:", file=f)
            print(model.summary(), file=f)
            print("-- Entrenamiento final", file=f)
            print('Test loss: {:0.4f}'.format(score[0]), file=f)
            print('Val loss: {:0.4f}'.format(score[1]), file=f)
            print('Val accuracy: {:0.4f}'.format(score[2]), file=f)
