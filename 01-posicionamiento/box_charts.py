import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
data_base_dir = script_dir + '/prediction_output/paper1/'


# Datos proporcionados
data = [
    {
        'name': 'M1',
        'dirname': 'M1-model1_paper',
        'data': {
            'CS1': 'track_straight_01_all_sensors.mbd_window_1_4_100_median.csv',
            'CS2': 'track_straight_01_all_sensors.mbd_window_3_4_100_median.csv',
            'CS3': 'track_straight_01_all_sensors.mbd_window_1_12_100_median.csv',
            'CS4': 'track_straight_01_all_sensors.mbd_window_3_12_100_median.csv',
            'CS5': 'track_straight_01_all_sensors.mbd_window_3_12_100_tss.csv',
        }
    },
    {
        'name': 'M2',
        'dirname': 'M2-model2_propio',
        'data': {
            'CS1': 'track_straight_01_all_sensors.mbd_window_1_4_100_median.csv',
            'CS2': 'track_straight_01_all_sensors.mbd_window_3_4_100_median.csv',
            'CS3': 'track_straight_01_all_sensors.mbd_window_1_12_100_median.csv',
            'CS4': 'track_straight_01_all_sensors.mbd_window_3_12_100_median.csv',
            'CS5': 'track_straight_01_all_sensors.mbd_window_3_12_100_tss.csv',
        }
    },
    {
        'name': 'M3',
        'dirname': 'M3-model3_extrainfo',
        'data': {
            'CS1': 'M3-model3_extrainfo_1_4_100_median.csv',
            'CS2': 'M3-model3_extrainfo_3_4_100_median.csv',
            'CS3': 'M3-model3_extrainfo_1_12_100_median.csv',
            'CS4': 'M3-model3_extrainfo_3_12_100_median.csv',
            'CS5': 'M3-model3_extrainfo_3_12_100_tss.csv',
        }
    },{
        'name': 'M4',
        'dirname': 'M4-model4_extrainfo_full',
        'data': {
            'CS1': 'M4-model4_extrainfo_full_1_4_100_median.csv',
            'CS2': 'M4-model4_extrainfo_full_3_4_100_median.csv',
            'CS3': 'M4-model4_extrainfo_full_1_12_100_median.csv',
            'CS4': 'M4-model4_extrainfo_full_3_12_100_median.csv',
            'CS5': 'M4-model4_extrainfo_full_3_12_100_tss.csv',
        }
    },
    {
        'name': 'M5',
        'dirname': 'M5-cnn',
        'data': {
            'CS1': 'track_straight_01_all_sensors.mbd_window_1_4_100_median.csv',
            'CS2': 'track_straight_01_all_sensors.mbd_window_3_4_100_median.csv',
            'CS3': 'track_straight_01_all_sensors.mbd_window_1_12_100_median.csv',
            'CS4': 'track_straight_01_all_sensors.mbd_window_3_12_100_median.csv',
            'CS5': 'track_straight_01_all_sensors.mbd_window_3_12_100_tss.csv',
        }
    },{
        'name': 'M6',
        'dirname': 'M6-rejilla_modelo1',
        'data': {
            'CS1': 'M6-rejilla_modelo1_1_4_100_median.csv',
            'CS2': 'M6-rejilla_modelo1_3_4_100_median.csv',
            'CS3': 'M6-rejilla_modelo1_1_12_100_median.csv',
            'CS4': 'M6-rejilla_modelo1_3_12_100_median.csv',
            'CS5': 'M6-rejilla_modelo1_3_12_100_tss.csv',
        }
    },{
        'name': 'M7',
        'dirname': 'M7-rejilla_modelo3',
        'data': {
            'CS1': 'M7-rejilla_modelo3_1_4_100_median.csv',
            'CS2': 'M7-rejilla_modelo3_3_4_100_median.csv',
            'CS3': 'M7-rejilla_modelo3_1_12_100_median.csv',
            'CS4': 'M7-rejilla_modelo3_3_12_100_median.csv',
            'CS5': 'M7-rejilla_modelo3_3_12_100_tss.csv',
        }
    },{
        'name': 'M8',
        'dirname': 'M8.2-rejilladinamica_modelo2_kerastuner',
        'data': {
            'CS1': 'M8.2-rejilladinamica_modelo2_kerastuner_1_4_100_median.csv',
            'CS2': 'M8.2-rejilladinamica_modelo2_kerastuner_3_4_100_median.csv',
            'CS3': 'M8.2-rejilladinamica_modelo2_kerastuner_1_12_100_median.csv',
            'CS4': 'M8.2-rejilladinamica_modelo2_kerastuner_3_12_100_median.csv',
            'CS5': 'M8.2-rejilladinamica_modelo2_kerastuner_3_12_100_tss.csv',
        }
    },
]

for item in data:
    name = item['name']
    dirname = item['dirname']
    data_values = item['data']

    model_values = {}
    for key, value in data_values.items():
        filepath = os.path.join(data_base_dir, dirname, value)
        df = pd.read_csv(filepath)
        euc_distance = df['eclidean_distance']
        model_values[key] = euc_distance

    model_amount_values = [len(x) for x in model_values.values()]

    # Plot
    fig, ax = plt.subplots()
    ax.boxplot(model_values.values())
    ax.set_xticklabels(model_values.keys())
    ax.yaxis.grid(True)  # Agregar líneas de guía horizontales
    ax.set_ylim([0, 16])
    plt.xlabel('capture settings')
    plt.ylabel('m')
    plt.title(name +' - Average deviation on euclidian distance')
    plt.savefig(os.path.join(data_base_dir+'charts/', name+'-boxplot.png'))
    plt.show()

    # Plot bar chart
    fig, ax = plt.subplots()
    ax.bar(model_values.keys(), model_amount_values)
    ax.yaxis.grid(True)  # Agregar líneas de guía horizontales
    plt.xlabel('capture settings')
    plt.ylabel('samples')
    plt.title('Samples used')
    plt.savefig(os.path.join(data_base_dir+'charts/', name+'-samples.png'))
    plt.show()

