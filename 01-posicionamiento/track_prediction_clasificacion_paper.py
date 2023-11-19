import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import autokeras as ak
import pickle
import random
import sys
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = script_dir+'/../'
sys.path.insert(1, root_dir)
from lib.trainingcommon import gridList_to_posXY, set_random_seed_value
from lib.trainingcommon import posXYlist_to_grid
from lib.trainingcommon import descale_pos_y
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import load_real_track_data

# Configuración
# 'track_1_rssi'#'track_straight_01_all_sensors.mbd_window_median'
track_file_prefix = 'track_straight_01_all_sensors.mbd_window'#'track_1_rssi'#'track_straight_01_all_sensors.mbd_window_median'
model_collection = 'clasificacion1'
dim_x = 20.660138018121128
dim_y = 17.64103475472807
cell_amount_x = 9
cell_amount_y = 9
random_seed = 42

# Configuración de los modelos a probar
modelList = [
    'M6-rejilla_modelo1',
    # 'M7-rejilla_modelo3',
]

# Configuración de las ventanas a predecir
windowsettingslist = [
    '1_4_100_median',
    '3_4_100_median',
    '1_12_100_median',
    '3_12_100_median',
    '3_12_100_tss'
]
# Cargamos la semilla de los generadores aleatorios
set_random_seed_value(random_seed)

for model_name in modelList:
    print("---- Modelo: "+model_name+" ----")
    model_deviation_data = []
    for windowsettings_suffix in windowsettingslist:
        print("---- Predicción de la ventana: "+windowsettings_suffix+" ----")

        # Rutas
        input_file_name = track_file_prefix+'_'+windowsettings_suffix
        output_file_name = model_name+'_'+windowsettings_suffix
        track_file = root_dir+'/preprocessed_inputs/paper1/'+input_file_name+'.csv'
        model_dir = script_dir+'/models/'+model_collection
        scaler_file = model_dir+'/files/paper1/' + model_name+'/scaler_'+windowsettings_suffix+'.pkl'
        model_file = model_dir+'/files/paper1/' + model_name+'/model_'+windowsettings_suffix+'.tf'
        output_dir = script_dir+'/prediction_output/paper1/'+model_name+'/'
        output_file = output_dir+output_file_name+'.csv'
        deviation_file = output_dir+output_file_name+'_deviations.csv'
        figure_file = output_dir+output_file_name+'.png'
        model_deviation_file = output_dir+'/'+track_file_prefix+'_deviations.csv'

        # Preparamos los datos
        input_data, output_data = load_real_track_data(
            track_file, scaler_file, False, False, True)
        output_data = output_data.to_numpy()
        output_data_grid = posXYlist_to_grid(
            output_data, cell_amount_x, cell_amount_y)

        # Cargamos el modelo
        model = tf.keras.models.load_model(
            model_file, custom_objects=ak.CUSTOM_OBJECTS)

        # Predecimos
        # tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.tables_initializer(name='init_all_tables'))
        predictions = model.predict(input_data)
        predictions = np.argmax(predictions, axis=-1)

        # Convertimos a posiciones
        predictions_positions = gridList_to_posXY(
            predictions, cell_amount_x, cell_amount_y)

        # print("-Predicciones-")
        # print("Real:")
        # print(output_data)
        # print("Estimado:")
        # print(predictions)

        # Desescalamos
        # with open(scaler_output_file, 'rb') as scalerFile:
        #  scaler = pickle.load(scalerFile)
        #  scalerFile.close()
        # predictions = scaler.inverse_transform(predictions)

        # Componemos la salida
        output_list = []
        for index in range(0, len(predictions)):
            listrow = {
                'real_x': output_data[index][0],
                'real_y': output_data[index][1],
                'real_grid': output_data_grid[index],
                'predicted_x': predictions_positions[index][0],
                'predicted_y': predictions_positions[index][1],
                'predicted_grid': predictions[index],
            }
            output_list.append(listrow)
        output_data = pd.DataFrame(output_list)

        # Preparamos cálculos
        output_data['deviation_x'] = (
            output_data['predicted_x'] - output_data['real_x']).abs()
        output_data['deviation_y'] = (
            output_data['predicted_y'] - output_data['real_y']).abs()
        output_data['eclidean_distance'] = np.sqrt(np.power(
            output_data['deviation_x'], 2) + np.power(output_data['deviation_y'], 2))
        output_data['deviation_grid'] = (
            output_data['predicted_grid'] - output_data['real_grid']).abs()

        # Imprimimos la desviacion máxima minima y media de X e Y
        print("- Desviaciones en predicciones -")
        print("Desviación máxima X: "+str(output_data['deviation_x'].max()))
        print("Desviación mínima X: "+str(output_data['deviation_x'].min()))
        print("Desviación media X: "+str(output_data['deviation_x'].mean()))
        print("Desviación X cuartil 25%: " +
              str(output_data['deviation_x'].quantile(0.25)))
        print("Desviación X cuartil 50%: " +
              str(output_data['deviation_x'].quantile(0.50)))
        print("Desviación X cuartil 75%: " +
              str(output_data['deviation_x'].quantile(0.75)))

        print("Desviación máxima Y: "+str(output_data['deviation_y'].max()))
        print("Desviación mínima Y: "+str(output_data['deviation_y'].min()))
        print("Desviación media Y: "+str(output_data['deviation_y'].mean()))
        print("Desviación Y cuartil 25%: " +
              str(output_data['deviation_y'].quantile(0.25)))
        print("Desviación Y cuartil 50%: " +
              str(output_data['deviation_y'].quantile(0.50)))
        print("Desviación Y cuartil 75%: " +
              str(output_data['deviation_y'].quantile(0.75)))

        print("Distancia euclídea máxima: " +
              str(output_data['eclidean_distance'].max()))
        print("Distancia euclídea mínima: " +
              str(output_data['eclidean_distance'].min()))
        print("Distancia euclídea media: " +
              str(output_data['eclidean_distance'].mean()))
        print("Desviación euclídea cuartil 25%: " +
              str(output_data['eclidean_distance'].quantile(0.25)))
        print("Desviación euclídea cuartil 50%: " +
              str(output_data['eclidean_distance'].quantile(0.50)))
        print("Desviación euclídea cuartil 75%: " +
              str(output_data['eclidean_distance'].quantile(0.75)))

        print("Desviación máxima rejilla: " +
              str(output_data['predicted_grid'].max()))
        print("Desviación media rejilla: " +
              str(output_data['predicted_grid'].mean()))
        print("Desviación mínima rejilla: " +
              str(output_data['predicted_grid'].min()))
        print("Desviación rejilla cuartil 25%: " +
              str(output_data['predicted_grid'].quantile(0.25)))
        print("Desviación rejilla cuartil 50%: " +
              str(output_data['predicted_grid'].quantile(0.50)))
        print("Desviación rejilla cuartil 75%: " +
              str(output_data['predicted_grid'].quantile(0.75)))

        # Guardamos las desviaciones en csv
        deviation_values = pd.DataFrame([{
            'min_x': output_data['deviation_x'].min(),
            'max_x': output_data['deviation_x'].max(),
            'mean_x': output_data['deviation_x'].mean(),
            'q25_x': output_data['deviation_x'].quantile(0.25),
            'q50_x': output_data['deviation_x'].quantile(0.50),
            'q75_x': output_data['deviation_x'].quantile(0.75),
            'min_y': output_data['deviation_y'].min(),
            'max_y': output_data['deviation_y'].max(),
            'mean_y': output_data['deviation_y'].mean(),
            'q25_y': output_data['deviation_y'].quantile(0.25),
            'q50_y': output_data['deviation_y'].quantile(0.50),
            'q75_y': output_data['deviation_y'].quantile(0.75),
            'min_euclidean': output_data['eclidean_distance'].min(),
            'max_euclidean': output_data['eclidean_distance'].max(),
            'mean_euclidean': output_data['eclidean_distance'].mean(),
            'q25_euclidean': output_data['eclidean_distance'].quantile(0.25),
            'q50_euclidean': output_data['eclidean_distance'].quantile(0.50),
            'q75_euclidean': output_data['eclidean_distance'].quantile(0.75),
        }])
        deviation_values.to_csv(deviation_file, index=False)

        # Hacemos la salida
        output_data.to_csv(output_file, index=False)

        # Mostramos el grafico
        plt.plot([0, 0, dim_x, dim_x, 0], [0, dim_y,  dim_y, 0, 0],
                 'go-', label='Real', linewidth=1)
        plt.plot(output_data['real_x'].values,
                 output_data['real_y'].values, 'ro-', label='Real', linewidth=1)
        plt.plot(output_data['predicted_x'].values,
                 output_data['predicted_y'].values, 'mo-', label='Calculada', linewidth=1)
        plt.savefig(figure_file)
        plt.close()

        # Guardamos los datos de desviación para el fichero de desviaciones global
        model_deviation_data.append({
            'windowsettings': windowsettings_suffix,
            'min_x': output_data['deviation_x'].min(),
            'max_x': output_data['deviation_x'].max(),
            'mean_x': output_data['deviation_x'].mean(),
            'q25_x': output_data['deviation_x'].quantile(0.25),
            'q50_x': output_data['deviation_x'].quantile(0.50),
            'q75_x': output_data['deviation_x'].quantile(0.75),
            'min_y': output_data['deviation_y'].min(),
            'max_y': output_data['deviation_y'].max(),
            'mean_y': output_data['deviation_y'].mean(),
            'q25_y': output_data['deviation_y'].quantile(0.25),
            'q50_y': output_data['deviation_y'].quantile(0.50),
            'q75_y': output_data['deviation_y'].quantile(0.75),
            'min_euclidean': output_data['eclidean_distance'].min(),
            'max_euclidean': output_data['eclidean_distance'].max(),
            'mean_euclidean': output_data['eclidean_distance'].mean(),
            'q25_euclidean': output_data['eclidean_distance'].quantile(0.25),
            'q50_euclidean': output_data['eclidean_distance'].quantile(0.50),
            'q75_euclidean': output_data['eclidean_distance'].quantile(0.75),
        })

    # Imprimimos los datos de desviación para el fichero de desviaciones global
    model_deviation_data = pd.DataFrame(model_deviation_data)
    model_deviation_data.to_csv(model_deviation_file, index=False)
