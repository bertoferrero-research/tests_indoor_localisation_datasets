
import sys
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir+'/../../../'  # Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from lib.trainingcommon import load_data
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import load_training_data
from lib.trainingcommon import save_model
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
import autokeras as ak
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split


# Variables globales

modelname = 'model4_extrainfo_full'
windowsettings_suffix = '1_4_100_median'
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
data_file = root_dir+'preprocessed_inputs/paper1/fingerprint_history_window_'+windowsettings_suffix+'.csv'
scaler_file = script_dir+'/files/paper1/'+modelname+'/scaler_'+windowsettings_suffix+'.pkl'
model_file = script_dir+'/files/paper1/'+modelname+'/model_'+windowsettings_suffix+'.tf'
model_image_file = script_dir+'/files/paper1/'+modelname+'/model_plot.png'
random_seed = 42

# Autokeras config
max_trials = 50
overwrite = True
autokeras_project_name = 'dense_modelo4'
auokeras_folder = root_dir+'/tmp/autokeras_training/'

# Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Construcción del modelo ---- #

# Cargamos los ficheros
X, y, Xmap = load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False,
                       scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

#Convertimos a numpy y formato
X = X.to_numpy()
y = y.to_numpy()
Xmap = Xmap.to_numpy()

# Construimos el modelo

# Entradas
inputSensors = ak.StructuredDataInput(name='input_sensors')
InputMap = ak.StructuredDataInput(name='input_map')

# Capas ocultas para cada entrada
hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors')(inputSensors)
hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map')(InputMap)

# Concatenamos las capas
concat = ak.Merge()([hiddenLayer_sensors, hiddenLayer_map])

# Capas ocultas tras la concatenación
hiddenLayer = ak.DenseBlock(use_batchnorm=False)(concat)

# Salida
output = ak.RegressionHead(metrics=['mse', 'accuracy'])(hiddenLayer)

# Construimos el modelo
model = ak.AutoModel(
    inputs=[inputSensors, InputMap],
    outputs=output, 
    overwrite=overwrite,
    seed=random_seed,
    max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder)

# Entrenamos
X_train, X_test, y_train, y_test, Xmap_train, Xmap_test = train_test_split(
    X, y, Xmap, test_size=0.2)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, restore_best_weights=True)
history = model.fit([X_train, Xmap_train], y_train, validation_data=([X_test, Xmap_test], y_test),
                    verbose=2, callbacks=[callback])

# Evaluamos usando el test set
score = model.evaluate([X_test, Xmap_test], y_test, verbose=0)


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

print("-- Entrenamiento final")
print('Test loss: {:0.4f}'.format(score[0]))
print('Val loss: {:0.4f}'.format(score[1]))
print('Val accuracy: {:0.4f}'.format(score[2]))


#Guardamos la imagen resumen
tf.keras.utils.plot_model(model, to_file=model_image_file, show_shapes=True, show_layer_names=False, show_dtype=False, show_layer_activations=False)

#plot_learning_curves(history)
#print(score)