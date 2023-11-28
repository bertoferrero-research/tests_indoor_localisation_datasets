import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
from sklearn.model_selection import train_test_split
import random
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../'                                                                        #Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from lib.trainingcommon import set_random_seed_value, load_data, save_history, save_model
from models.M1 import M1

# -- Configuración -- #

# Variables globales
random_seed = 42
batch_size = 256

#Configuración de la prueba
test_values = list(range(1, 13))
test_name = 'FST1'
ouput_dir = script_dir+'/output/'+test_name+'/models/'
input_data_dir = root_dir+'preprocessed_inputs/paper1/'
training_data_filename = input_data_dir+test_name+'-fingerprint_history_window_1_-test_variable-_100_median.csv'
test_data_filename = input_data_dir+test_name+'-FST1-track_straight_01_all_sensors.mbd_window_1_-test_variable-_100_median.csv'
scaler_filename = ouput_dir+'-test_variable--scaler.pkl'
model_filename = ouput_dir+'-test_variable--model.keras'
history_filename = ouput_dir+'-test_variable--history.pkl'

# -- END Configuración -- #

# Cargamos la semilla de los generadores aleatorios
set_random_seed_value(random_seed)

# Recorremos cada valor de test_values
for test_value in test_values:
	#Definimos ficheros
	training_data_filename_execution = training_data_filename.replace('-test_variable-', str(test_value))
	test_data_filename_execution = test_data_filename.replace('-test_variable-', str(test_value))
	scaler_filename_execution = scaler_filename.replace('-test_variable-', str(test_value))
	model_filename_execution = model_filename.replace('-test_variable-', str(test_value))
	history_filename_execution = history_filename.replace('-test_variable-', str(test_value))

	#Cargamos los datos
	X_train, y_train = load_data(training_data_filename_execution, scaler_filename_execution, train_scaler_file=True, include_pos_z=False, scale_y=True, remove_not_full_rows=False)


	#Construimos el modelo
	inputlength = X_train.shape[1]
	outputlength = y_train.shape[1]
	model = M1(inputlength=inputlength, outputlength=outputlength).build_model()

	#Entrenamos
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
	history = model.fit(X_train, y_train, epochs=1000, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback], verbose=1)

	#Guardamos el modelo y el histórico
	save_model(model, model_filename_execution)
	save_history(history, history_filename_execution)