import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os
import os.path
# Referencia al directorio actual, por si ejecutamos el python en otro directorio
script_dir = os.path.dirname(os.path.abspath(__file__))
data_base_dir = script_dir + '/paper1/output/'


# Datos proporcionados
data = [
    {
        'name': 'T1 - Closer sensors',
        'dirname': 'T1_closer_sensors',
        'data': {
            'CS1': 'track_straight_01_all_sensors.mbd_window_1_4_100_median.csv',
            'CS2': 'track_straight_01_all_sensors.mbd_window_3_4_100_median.csv',
            'CS3': 'track_straight_01_all_sensors.mbd_window_1_12_100_median.csv',
            'CS4': 'track_straight_01_all_sensors.mbd_window_3_12_100_median.csv',
            #'CS5': 'track_straight_01_all_sensors.mbd_window_3_12_100_tss.csv',
        }
    },
    {
        'name': 'T2 - Static sensors',
        'dirname': 'T2_static_sensors',
        'data': {
            'CS1': 'track_straight_01_all_sensors.mbd_window_1_4_100_median.csv',
            'CS2': 'track_straight_01_all_sensors.mbd_window_3_4_100_median.csv',
            'CS3': 'track_straight_01_all_sensors.mbd_window_1_12_100_median.csv',
            'CS4': 'track_straight_01_all_sensors.mbd_window_3_12_100_median.csv',
            #'CS5': 'track_straight_01_all_sensors.mbd_window_3_12_100_tss.csv',
        }
    }
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
    ax.set_ylim([0, 20])
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
    plt.title(name +' - Samples used')
    plt.savefig(os.path.join(data_base_dir+'charts/', name+'-samples.png'))
    plt.show()


