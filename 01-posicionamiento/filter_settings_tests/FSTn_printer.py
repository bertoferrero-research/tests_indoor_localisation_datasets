import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import numbers
import pickle
import autokeras as ak
from sklearn.model_selection import train_test_split
import random
import sys
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
root_dir = script_dir+'/../../'                                                                        #Referencia al directorio raiz del proyecto
sys.path.insert(1, root_dir)
from lib.trainingcommon import set_random_seed_value, load_data, save_history, save_model, descale_pos_x, descale_pos_y
from lib.models import M1

# Objetivos:
# FST1: Nº mínimo de medidas por sensor obligatorias VS error de predicción y número de muestras
# FST2: Nº mínimo de sensores con valor obligatorio VS error de predicción y número de muestras
# FST3: Tipo de filtrado de los sensores VS error de predicción y número de muestras
# FST4: Tamaño máximo de ventana VS error de predicción y número de muestras
# FST5: Tamaño mínimo de ventana VS error de predicción y número de muestras
# -- Configuración -- #

# Configuración por test
test_specific_settings = {
	'FST1': {
		'chart_x_label': 'Minimal required measures per sensor',
		'test_values': [1, 2, 3, 4, 5, 6],
	},
	'FST2': {
		'chart_x_label': 'Minimal required sensors without empty values',
		'test_values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
	},
	'FST3': {
		'chart_x_label': 'Filter type',
		'test_values': ['max', 'min', 'mean', 'median', 'TSS'],
	},
	'FST4': {
		'chart_x_label': 'Max window size (sec.)',
		'test_values': ["1.0", "1.25", "1.5", "1.75", "2.0", "2.25", "2.5", "2.75", "3.0"],
	},
	'FST5': {
		'chart_x_label': 'Min window size (sec.)',
		'test_values': ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75"],
	},
}

# Variables globales
random_seed = 42
batch_size = 256
dim_x = 20.660138018121128
dim_y = 17.64103475472807

#Configuración de la prueba
test_name = 'FST5'
output_dir = script_dir+'/output/'+test_name+'/'
output_dir_models = output_dir+'models/'
input_data_dir = root_dir+'preprocessed_inputs/paper1/'
training_data_filename = input_data_dir+test_name+'-fingerprint_-test_variable-.csv'
test_data_filename = input_data_dir+test_name+'-track_-test_variable-.csv'
scaler_filename = output_dir_models+'-test_variable--scaler.pkl'
model_filename = output_dir_models+'-test_variable--model.keras'
history_filename = output_dir_models+'-test_variable--history.pkl'

#Salida de los datos de test
test_output_file = output_dir+'/predictions/'+test_name+'_predictions_-test_variable-.csv'
general_figure_file = output_dir+'results.png'
general_boxfigure_file = output_dir+test_name+'_result'
general_samples_frecuency_file = output_dir+'samples_frecuency.png'
general_results_file = output_dir+'results_data.csv'
plot_output_extensions = ['.png', '.eps']

# -- END Configuración -- #

# Extreamos la configuración específica de la prueba
chart_x_label = test_specific_settings[test_name]['chart_x_label']
test_values = test_specific_settings[test_name]['test_values']

# Cargamos la semilla de los generadores aleatorios
set_random_seed_value(random_seed)

# Cargamos datos desde csv
deviations = []
for test_value in test_values:
	test_output_file_execution = test_output_file.replace('-test_variable-', str(test_value))
	test_data = pd.read_csv(test_output_file_execution)
	deviations.append(test_data['eclidean_distance'])
general_results = pd.read_csv(general_results_file)

##Imprimimos la gráfica de cajas
# Primer eje con el error medio
fig, ax1 = plt.subplots()
ax1.set_xlabel(chart_x_label)
ax1.set_ylabel('Euclidian distance error (m)')
ax1.set_ylim([0, 12])

#Convertimos test_value a string para que no haya problemas en el eje x
general_results['test_value'] = general_results['test_value'].astype(str)

# Calculate the positions for the boxplots
positions = np.arange(len(general_results['test_value']))
plot_1 = ax1.boxplot(deviations, positions=positions, showfliers=True, widths=0.2, showmeans=True, meanline=True, notch=True)
ax1.set_xticklabels(general_results['test_value'])

# Segundo eje con la cantidad de muestras
ax2 = ax1.twinx()
#ax2.set_ylabel('Samples amount', color='tab:red')
ax2.set_ylim([0, 70])
plot_2 = ax2.plot(general_results['test_value'], general_results['test_samples_amount'], label='Samples amount', color='tab:red', marker='o')
#ax2.tick_params(axis='y', labelcolor='tab:red')
#ax2.set_xticklabels(general_results['test_value'])
ax2.set_yticklabels([]) #Ocultar valores en el eje
ax2.yaxis.grid(False)

# Iterar sobre las posiciones y los números de muestras
for pos, num_samples in zip(positions, general_results['test_samples_amount']):
    # Usar annotate para agregar una anotación en la posición correspondiente
    ax1.annotate(str(num_samples), (pos, 1), xytext=(0, 250), 
                 textcoords='offset points', ha='center', va='bottom', color='tab:red')
ax1.annotate("Samples amount", (0.5, 1), xytext=(0, -20), xycoords='figure fraction', textcoords='offset points', ha='center', va='bottom', color='tab:red')


# Tercer eje con la frecuencia de muestras
ax3 = ax1.twinx()
ax3.set_xticks(positions)
ax3.set_ylabel('Time between samples (sec.)', color='tab:blue')
ax3.set_ylim([0, 3])
plot_3 = ax3.plot(general_results['test_value'], general_results['mean_time_diff'], label='Time between samples (sec.)', color='tab:blue', marker='o')
ax3.tick_params(axis='y', labelcolor='tab:blue')
ax3.set_xticklabels(general_results['test_value'])

plt.xticks(general_results['test_value'])
plt.grid(True)
for extension in plot_output_extensions:
	plt.savefig(general_boxfigure_file+extension)


# ##Tercer gráfico con la frecuencia de muestras
# fig, ax1 = plt.subplots()
# ax1.set_xlabel(chart_x_label)
# ax1.set_ylabel('Time difference between samples (sec.)')
# ax1.set_ylim([0, 6])

# # Calculate the positions for the boxplots
# positions = np.arange(1, len(general_results['test_value'])+1)
# plot_1 = ax1.boxplot(time_diffs, positions=positions, showfliers=True, widths=0.2)
# ax1.set_xticklabels(general_results['test_value'])

# # Segundo eje con la cantidad de muestras
# ax2 = ax1.twinx()
# ax2.set_ylabel('Samples amount', color='tab:red')
# ax2.set_ylim([0, 70])
# plot_2 = ax2.plot(general_results['test_value'], general_results['test_samples_amount'], label='Samples amount', color='tab:red', marker='o')
# ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.set_xticklabels(general_results['test_value'])

# plt.xticks(general_results['test_value'])
# plt.grid(True)
# plt.savefig(general_samples_frecuency_file)