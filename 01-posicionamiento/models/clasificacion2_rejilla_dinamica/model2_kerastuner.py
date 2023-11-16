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
import keras_tuner
from sklearn.model_selection import train_test_split
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import plot_learning_curves
from lib.trainingcommon import load_training_data
from lib.trainingcommon import posXYList_to_dinamic_grid,posXYlist_to_grid
from lib.trainingcommon import descale_dataframe
from lib.trainingcommon import save_model
from lib.trainingcommon import load_data, set_random_seed_value

# -- Configuración específica -- #
cell_amount_x = 3
cell_amount_y = 3

# -- Configuración -- #

# Variables globales y configuración
modelname = 'M8.2-rejilladinamica_modelo2_kerastuner'
random_seed = 42
# Keras config
use_gpu = True
# Kerastuner config
overwrite = True
tuner = 'bayesian'
batch_size = 256
loss = 'categorical_crossentropy' #'mse'
max_trials = 50

#Configuración de las ventanas a usar
windowsettingslist = [
  '1_4_100_median',
  '3_4_100_median',
  '1_12_100_median',
  '3_12_100_median',
  '3_12_100_tss'
]

# -- END Configuración -- #
 

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
    model_image_file = script_dir+'/files/paper1/'+modelname+'/model_plot.png'    
    autokeras_project_name = modelname
    auokeras_folder = root_dir+'/tmp/autokeras_training/'

    # ---- Construcción del modelo ---- #

    #Cargamos los ficheros
    X, y = load_training_data(data_file, scaler_file, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
    #track_X, track_y = load_data(track_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=False, remove_not_full_rows=True)
    #X = pd.concat([X, track_X])
    #y = pd.concat([y, track_y])

    #Cargamos cada dimension
    y_dim1 = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)
    y_dim2 = posXYlist_to_grid(y.to_numpy(), cell_amount_x**2, cell_amount_y**2)

    #Convertimos a categorical
    y_dim1 = tf.keras.utils.to_categorical(y_dim1, num_classes=cell_amount_x*cell_amount_y)
    y_dim2 = tf.keras.utils.to_categorical(y_dim2, num_classes=(cell_amount_x**2)*(cell_amount_y**2))


    #Construimos el modelo    
    inputlength = X.shape[1]
    outputlength_dim1 = y_dim1.shape[1]
    outputlength_dim2 = y_dim2.shape[1]
    def myhypermodel(hp):

        #Parte 1 - Dimension 1
        input_rssi = tf.keras.Input(shape=(inputlength,), name='input_rssi')
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_input', 0.0, 0.5, step=0.1, default=0.0))(input_rssi)
        hiddenLayers_d1 = tf.keras.layers.Dense(1024, activation='relu')(hiddenLayers_d1)
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_1.2', min_value=0.0, max_value=0.5, step=0.1, default=0.0))(hiddenLayers_d1)
        hiddenLayers_d1 = tf.keras.layers.Dense(256, activation='relu')(hiddenLayers_d1)
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_1.3', min_value=0.0, max_value=0.5, step=0.1, default=0.0))(hiddenLayers_d1)
        hiddenLayers_d1 = tf.keras.layers.Dense(1024, activation='relu')(hiddenLayers_d1)
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_1.4', min_value=0.0, max_value=0.5, step=0.1, default=0.0))(hiddenLayers_d1)
        output_d1 = tf.keras.layers.Dense(units=outputlength_dim1, activation='softmax', name='output_d1')(hiddenLayers_d1)

        #Parte 2 - Dimension 2
        concatenate_input_d2 = tf.keras.layers.Concatenate()([input_rssi, output_d1])
        hiddenLayers_d2 = tf.keras.layers.Dense(32, activation='relu')(concatenate_input_d2)
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_2.1', min_value=0.0, max_value=0.5, step=0.1, default=0.0))(hiddenLayers_d1)
        hiddenLayers_d2 = tf.keras.layers.Dense(16, activation='relu')(hiddenLayers_d2)
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_2.2', min_value=0.0, max_value=0.5, step=0.1, default=0.0))(hiddenLayers_d1)
        hiddenLayers_d2 = tf.keras.layers.Dense(1024, activation='relu')(hiddenLayers_d2)
        hiddenLayers_d1 = tf.keras.layers.Dropout(hp.Float('dropout_2.3', min_value=0.0, max_value=0.5, step=0.1, default=0.0))(hiddenLayers_d1)
        output_d2 = tf.keras.layers.Dense(outputlength_dim2, activation='softmax')(hiddenLayers_d2)

        model = tf.keras.Model(
            inputs=[input_rssi],
            outputs=[output_d1, output_d2]
        )
        #Optimizador        
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=0.00001, max_value=0.1, step=10, sampling="log"))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'], loss_weights=[0.2, 0.8] ) 

        return model

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)


    X_train, X_test, y_dim1_train, y_dim1_test, y_dim2_train, y_dim2_test = train_test_split(X, y_dim1, y_dim2, test_size=0.2)
    tuner = keras_tuner.BayesianOptimization(
        myhypermodel,
        objective="val_loss",
        max_trials=max_trials,
        overwrite=overwrite,
        directory=auokeras_folder,
        project_name=autokeras_project_name,
        seed=random_seed
    )

    tuner.search(X_train, [y_dim1_train, y_dim2_train], epochs=1000, validation_data=(X_test, [y_dim1_test, y_dim2_test]), 
                     verbose=2,
                     batch_size=batch_size,
                     callbacks=[callback])

    model = tuner.get_best_models()[0]
    model.build(input_shape=(inputlength,))
    # Evaluamos usando el test set
    score = model.evaluate(X_test, [y_dim1_test, y_dim2_test], verbose=0)


    #Guardamos el modelo
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
    tf.keras.utils.plot_model(model, to_file=model_image_file, show_shapes=True, show_layer_names=False, show_dtype=False, show_layer_activations=False)

    #plot_learning_curves(history)
    #print(score)