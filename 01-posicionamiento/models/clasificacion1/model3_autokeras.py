import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
import autokeras as ak
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import posXYlist_to_grid
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import save_model
from lib.trainingcommon import load_data, save_history, set_random_seed_value

# -- Configuración específica -- #
cell_amount_x = 7
cell_amount_y = 6

# -- Configuración -- #

# Variables globales y configuración
modelname = 'M7-rejilla_modelo3'
random_seed = 42
training_to_design = False #Indica si estamos entrenando el modelo para diseñarlo o para evaluarlo
# Keras config
use_gpu = True
# Autokeras config
max_trials = 50
overwrite = True
tuner = 'bayesian'
batch_size = 256

#Configuración de las ventanas a usar
windowsettingslist = [
  '1_4_100_median',
  '3_4_100_median',
  '1_12_100_median',
  '3_12_100_median',
  '3_12_100_tss'
]

# -- END Configuración -- #
 
#Si entrenamos para diseño, solo usamos una ventana
if training_to_design:
    windowsettingslist = [windowsettingslist[0]]

# Cargamos la semilla de los generadores aleatorios
set_random_seed_value(random_seed)

#Si no usamos GPU forzamos a usar CPU
if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    

# ---- Entrenamiento del modelo ---- #

#Recorrido de las configuraciones de ventana
for windowsettings_suffix in windowsettingslist:

    print("---- Entrenamiento del modelo ----")
    print("Configuración de la ventana: "+windowsettings_suffix)


    #Rutas
    data_file = root_dir+'preprocessed_inputs/paper1/fingerprint_history_window_'+windowsettings_suffix+'.csv'
    scaler_file = script_dir+'/files/paper1/'+modelname+'/scaler_'+windowsettings_suffix+'.pkl'
    model_file = script_dir+'/files/paper1/'+modelname+'/model_'+windowsettings_suffix+'.tf'
    history_file = script_dir+'/files/paper1/'+modelname+'/history_'+windowsettings_suffix+'.pkl'
    model_image_file = script_dir+'/files/paper1/'+modelname+'/model_plot.png'
    autokeras_project_name = modelname
    auokeras_folder = root_dir+'/tmp/autokeras_training/'

    # ---- Construcción del modelo ---- #

    #Cargamos los ficheros
    X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=False, remove_not_full_rows=True)

    y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)

    #Convertimos a categorical
    y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)


    #Construimos el modelo
    input = ak.StructuredDataInput()
    if training_to_design:
        layer = ak.DenseBlock(use_batchnorm=False)(input)
    else:
        layer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=256)(input)
        layer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=512)(input)
    output_layer = ak.ClassificationHead(metrics=['mse', 'accuracy'])(layer)

    model = ak.AutoModel(
        inputs=input,
        outputs=output_layer,
        overwrite=overwrite,
        objective = 'val_accuracy',
        tuner=tuner,
        seed=random_seed,
        max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder
    )

    #Entrenamos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        verbose=(1 if training_to_design else 2), callbacks=[callback], batch_size=batch_size)

    # Evaluamos usando el test set
    score = model.evaluate(X_test, y_test, verbose=0)


    #Guardamos el modelo
    model = model.export_model()
    save_model(model, model_file)

    #Sacamos valoraciones
    print("-- Resumen del modelo:")
    print(model.summary())

    # print("-- Evaluación cruzada")
    # print("Puntuaciones de validación cruzada:", cross_val_scores)
    # print("Puntuación media:", cross_val_scores.mean())
    # print("Desviación estándar:", cross_val_scores.std())

    print(score)
    print("-- Entrenamiento final")
    print('Test loss: {:0.4f}'.format(score[0]))
    print('Val loss: {:0.4f}'.format(score[1]))
    print('Val accuracy: {:0.4f}'.format(score[2]))


    #Guardamos la imagen resumen
    #tf.keras.utils.plot_model(model, to_file=model_image_file, show_shapes=True, show_layer_names=False, show_dtype=False, show_layer_activations=False)

    #plot_learning_curves(history)
    #print(score)

    overwrite = True