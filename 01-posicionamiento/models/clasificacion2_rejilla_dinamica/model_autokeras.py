import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import random
import shutil
import autokeras as ak
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import posXYList_to_dinamic_grid
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import save_model
from lib.trainingcommon import load_data


#Variables globales
data_file = script_dir+'/../../../preprocessed_inputs/fingerprint_history_window_median.csv'
#track_file = script_dir+'/../../../preprocessed_inputs/synthetic_tracks/track_1_rssi_12h.csv'
scaler_file = script_dir+'/files/scaler_autokeras.pkl'
model_file = script_dir+'/files/model_autokeras'
model_image_file = script_dir+'/output/model_plot.png'
random_seed = 42
cell_amount_x = 3
cell_amount_y = 3
zoom_level = 2

#Autokeras config
max_trials = 50
autokeras_project_name = 'posicionamiento_clasificacion2'
auokeras_folder = root_dir+'/tmp/autokeras_training/'

#Cargamos la semilla de los generadores aleatorios
np.random.seed(random_seed)
random.seed(random_seed)

# ---- Construcción del modelo ---- #

#Cargamos los ficheros
X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
#track_X, track_y = load_data(track_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
#X = pd.concat([X, track_X])
#y = pd.concat([y, track_y])
y = posXYList_to_dinamic_grid(y.to_numpy(), zoom_level, cell_amount_x, cell_amount_y)

#Separamos por dimension
y_dim1 = y[:,0]
y_dim2 = y[:,1]

#Convertimos a categorical
y_dim1 = tf.keras.utils.to_categorical(y_dim1, num_classes=cell_amount_x*cell_amount_y)
y_dim2 = tf.keras.utils.to_categorical(y_dim2, num_classes=cell_amount_x*cell_amount_y)


#Construimos el modelo
inputlength = X.shape[1]
outputlength_dim1 = y_dim1.shape[1]
outputlength_dim2 = y_dim2.shape[1]

#Parte 1 - Dimension 1
input_rssi = ak.StructuredDataInput(name='input_rssi')
hiddenLayers_d1 = ak.DenseBlock(use_batchnorm=False, name='hidden_layers_d1_1')(input_rssi)
output_d1 = ak.ClassificationHead(num_classes=outputlength_dim1, multi_label=False, name='output_d1')(hiddenLayers_d1)

#Parte 2 - Dimension 2
concatenate_input_d2 = ak.Merge(name='concatenate_input_d2')([input_rssi, output_d1])
hiddenLayers_d2 = ak.DenseBlock(use_batchnorm=False, name='hidden_layers_d2_1')(concatenate_input_d2)
output_d2 = ak.ClassificationHead(num_classes=outputlength_dim1, multi_label=False, name='output_d2')(hiddenLayers_d2)


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, restore_best_weights=True)
model = ak.AutoModel(inputs=input_rssi, outputs=[output_d1, output_d2],
    overwrite=False,
    #objective = 'val_output_d1_accuracy',
    max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder)

#Entrenamos
X_train, X_test, y_dim1_train, y_dim1_test, y_dim2_train, y_dim2_test = train_test_split(X, y_dim1, y_dim2, test_size=0.2)
history = model.fit(X_train, [y_dim1_train, y_dim2_train], validation_data=(X_test, [y_dim1_test, y_dim2_test]),
                     #batch_size=  batch_size,
                     #epochs=  epochs, 
                     verbose=1, callbacks=[callback])

# Evaluamos usando el test set
score = model.evaluate(X_test, [y_dim1_test, y_dim2_test], verbose=0)


#Guardamos el modelo
model = model.export_model()
save_model(model, model_file)

#Sacamos valoraciones
print("-- Resumen del modelo:")
print(model.summary())

print("-- Entrenamiento")
print('Test loss: {:0.4f}'.format(score[0]))

#Guardamos la imagen resumen
tf.keras.utils.plot_model(model, to_file=model_image_file, show_shapes=True, show_layer_names=False, show_dtype=False, show_layer_activations=False)

#plot_learning_curves(history)

# print(model.get_config())
# print(model.optimizer.get_config())
# print(model.loss)
# print(model.optimizer)
# print(model.history.params)