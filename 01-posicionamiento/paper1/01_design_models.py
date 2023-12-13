
import sys
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir+'/../../'  # Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from sklearn.model_selection import train_test_split
from lib.models.trainers import M2Trainer, M3Trainer, M4Trainer, M5Trainer
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
    # 'M1', #M1 no es necesario, ya está diseñado
    #'M2'
    #'M3', #Pendiente
    #'M4', #Pendiente
    'M5'
]

# Dataset a emplear
dataset = 'minsensors_10'

# Configuración del entrenamiento
batch_size = 256
max_trials = 50
overwrite = True
tuner = 'bayesian'

# Configuración general
random_seed = 42
models_dir = script_dir+'/models/'
scaler_filename = 'scaler.pkl'
model_filename = 'model.tf'
design_image_filename = 'model_plot.png'
tmp_dirname = root_dir+'/tmp/autokeras_training/'
dataset_path = root_dir+'preprocessed_inputs/paper1/dataset-fingerprint_'+dataset+'.csv'


# Configramos la semilla aleatoria
set_random_seed_value(random_seed)

# Recorremos cada modelo
for modelName in models:
    print('---- Diseñando modelo '+modelName + ' ----')
    # Definimos rutas
    model_dir = models_dir+modelName+'/'
    model_file = model_dir+model_filename
    design_image_file = model_dir+design_image_filename
    scaler_file = model_dir+scaler_filename
    model_tmp_dir = tmp_dirname+modelName+'/'
    log_file = model_dir + 'log.txt'  # Ruta del archivo de registro

    # Obtenemos la clase del modelo entrenador
    trainerClass = globals()[modelName+'Trainer']


    # Guardar el log de tensorflow en un archivo
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        open(log_file, "w").close()
    tf.get_logger().addHandler(logging.FileHandler(log_file, 'a'))

    # Mandamos entrenar el modelo
    model, score = trainerClass.train_model(dataset_path=dataset_path, scaler_file=scaler_file, tuner=tuner, tmp_dir=model_tmp_dir,
                                         batch_size=batch_size, designing=True, overwrite=overwrite, max_trials=max_trials, random_seed=random_seed)

    # Guardamos el modelo
    save_model(model, model_file)

    # Guardamos la imagen del diseño
    tf.keras.utils.plot_model(model, to_file=design_image_file, show_layer_names=True, show_shapes=True)

    # Imprimimos resultados
    with open(log_file, 'a') as f:
        print("-- Resumen del modelo:", file=f)
        print(model.summary(), file=f)
        print("-- Entrenamiento final", file=f)
        print('Test loss: {:0.4f}'.format(score[0]), file=f)
        print('Val loss: {:0.4f}'.format(score[1]), file=f)
        print('Val accuracy: {:0.4f}'.format(score[2]), file=f)
