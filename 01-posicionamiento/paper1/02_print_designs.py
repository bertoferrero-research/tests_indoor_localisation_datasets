
import sys
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir+'/../../'  # Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from sklearn.model_selection import train_test_split
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
    # 'M4', 
    # 'M5', 
    # 'M6', 
    # 'M7',
    'M8'
]

# Configuración general
models_dir = script_dir+'/models/'
model_filename = 'model.tf'
design_image_filename = 'model_plot.png'

# Configramos la semilla aleatoria
set_random_seed_value()

# Recorremos cada modelo
for modelName in models:
    # Definimos rutas
    model_dir = models_dir+modelName+'/'
    model_file = model_dir+model_filename
    design_image_file = model_dir+design_image_filename
    log_file = model_dir + 'log.txt'  # Ruta del archivo de registro

    # Si n oexiste el model_file (el cual es un directorio), pasamos
    if not os.path.isfile(model_file) and not os.path.exists(model_file):
        continue

    #Cargamos el modelo
    model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS, compile=False)
    model.compile()

    # Guardamos la imagen del diseño
    #No podemos guardar el diseño en imagen en la maquina de entrenamiento porque no podemos instalar la libreria
    tf.keras.utils.plot_model(model, to_file=design_image_file, show_layer_names=True, show_shapes=True)
