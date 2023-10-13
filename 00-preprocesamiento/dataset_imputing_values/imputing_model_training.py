import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, UpSampling1D, Input
script_dir = os.path.dirname(os.path.abspath(__file__)) #Referencia al directorio actual, por si ejecutamos el python en otro directorio
sys.path.insert(1, script_dir+'/../')
from lib.trainingcommon import plot_learning_curves


input_data = script_dir+'/../dataset_processed_csv/fingerprint_history_train.csv'
model_file = script_dir+'/files/model.h5'
output_csv = script_dir+'/files/test_output.csv'
empty_value = -200              #Valor que se usará para simular los valores vacios
max_simulated_empty_values = 3  #Número máximo de sensores que se pueden alterar por fila(empezará por 1 hasta alcanzar este numero, inclusive)
samples_per_row = 3             #Número de muestras a generar por fila

#Cargamos el fichero
df = pd.read_csv(input_data)
#Nos quedamos con las columnas de los sensores
df = df.iloc[:, 4:]
#Ordenamos alfabéticamente las columnas asegurandonos de que todos los datasets van en el mismo orden
df = df.reindex(sorted(df.columns), axis=1)
#Retenemos los nombres de los sensores
sensors_names = df.columns
#Reemplazamos el valor de vacío por nan
df = df.replace(empty_value, np.nan)
#Eliminamos las filas con nan
df = df.dropna()
#Creamos una copia como datos de salida
df_input = []
df_output = []
#simulamos las entradas vacias de tantos grupos como esté configurado
for i in range(1, max_simulated_empty_values+1):
    #De cada fila, alteramos el valor de tantos sensores como valor tenga la variable i
    for index, row in df.iterrows():
        for j in range(0, samples_per_row):
            input_row = row.to_dict()
            output_row = input_row.copy()
            #Obtenemos los sensores que vamos a alterar
            sensors_replace = np.random.choice(df.columns, i, replace=False)
            #Alteramos los sensores
            for sensor in sensors_replace:
                input_row[sensor] = empty_value
            #Añadimos la fila a los datos de entrada y salida
            df_input.append(input_row)
            df_output.append(output_row)

#Convertimos los datos a dataframe
df_input = pd.DataFrame(df_input)
df_output = pd.DataFrame(df_output)

#Mezclamos
np.random.seed(42)
indices_aleatorios = np.random.permutation(len(df_input))
df_input = df_input.iloc[indices_aleatorios].reset_index(drop=True)
df_output = df_output.iloc[indices_aleatorios].reset_index(drop=True)

#Normalizamos los datos para que estén entre 0 y 1 y usar la función de activación sigmoidal
scaler = MinMaxScaler()
scaler.fit([[-100], [0]])
for sensor in sensors_names:
    df_input[sensor] = scaler.transform(df_input[sensor].values.reshape(-1, 1)).flatten()
    df_output[sensor] = scaler.transform(df_output[sensor].values.reshape(-1, 1)).flatten()

##Creamos el modelo
#Dense
#model = Sequential()
'''
input_sensors = Input(shape=(df_input.shape[1]))
encoded = Dense(8, activation='relu')(input_sensors)
encoded = Dense(7, activation='relu')(input_sensors)
encoded = Dense(6, activation='relu')(input_sensors)
encoded = Dense(5, activation='relu')(input_sensors)
encoded = Dense(4, activation='relu')(input_sensors)
encoded = Dense(3, activation='relu')(input_sensors)
encoded = Dense(4, activation='relu')(input_sensors)
encoded = Dense(5, activation='relu')(input_sensors)
encoded = Dense(6, activation='relu')(input_sensors)
encoded = Dense(7, activation='relu')(input_sensors)
encoded = Dense(8, activation='relu')(input_sensors)
decoded = Dense(df_output.shape[1], activation='linear')(encoded)
'''


#CNN
# Entrada
input_sensors = Input(shape=(df_input.shape[1], 1)) 
# Encoder...
x = Conv1D(64, 3, activation='relu', padding='same')(input_sensors)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(128, 2, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(256, 2, activation='relu', padding='same')(x)
encoded = MaxPooling1D(1, padding='same')(x)

# Decoder...
x = Conv1D(256, 2, activation='relu', padding='same')(encoded)
x = UpSampling1D(1)(x)
x = Conv1D(128, 2, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(64, 3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)


# Capa de salida con 1 convolución
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)


'''
model = Sequential()
#Encoder
model.add(Conv1D(64, 2, activation='relu', input_shape=(df_input.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 2, activation='relu'))
model.add(MaxPooling1D(2))
#Decoder
model.add(Conv1D(32, 2, activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(64, 2, activation='relu'))
model.add(UpSampling1D(2))
#model.add(Conv1D(1, 2, activation='relu'))
'''

#Compilamos el modelo
model = Model(input_sensors, decoded)
model.compile(optimizer='RMSProp', loss='mae', metrics=['accuracy','mse','mae'])

# Resumen del modelo
model.summary()

#Creamos grupo de entrenamiento y test
#Dividimos el dataset en 80% para entrenamiento y 20% para test
train_size = int(len(df_input) * 0.8)
train_input, test_input = df_input.iloc[:train_size].reset_index(drop=True), df_input.iloc[train_size:].reset_index(drop=True)
train_output, test_output = df_output.iloc[:train_size].reset_index(drop=True), df_output.iloc[train_size:].reset_index(drop=True)

#Entrenamos el modelo
train_input = train_input.values.reshape(train_input.shape[0], train_input.shape[1], 1)
train_output = train_output.values.reshape(train_output.shape[0], train_output.shape[1], 1)
history = model.fit(train_input, train_output, epochs=40, batch_size=1250, validation_data=(test_input, test_output))


plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate(test_input, test_output, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

#Guardamos el modelo
model.save(model_file)

#Realizamos pruebas con test_input, estimando cada fila y guardando el resultado con test_output en un csv
test_input = test_input.values.reshape(test_input.shape[0], test_input.shape[1], 1)
test_predicted = model.predict(test_input)
test_predicted = test_predicted.reshape(test_predicted.shape[0], test_predicted.shape[1])
test_input = test_input.reshape(test_input.shape[0], test_input.shape[1])

#Desnormalizamos los datos
test_input = scaler.inverse_transform(test_input)
test_predicted = scaler.inverse_transform(test_predicted)
test_output = scaler.inverse_transform(test_output)

#Convertimos a dataframe los elementos que han sido alterados por reshape
test_input = pd.DataFrame(test_input, columns=sensors_names)
test_predicted = pd.DataFrame(test_predicted, columns=sensors_names).round()
test_output = pd.DataFrame(test_output, columns=sensors_names)

#Cambiamos el nombre de las columnas de los distintos dataframes
test_predicted.columns = [str(col) + '_predicted' for col in test_predicted.columns]
test_input.columns = [str(col) + '_input' for col in test_input.columns]
test_output.columns = [str(col) + '_output' for col in test_output.columns]
#Concatenamos test_input y test_predicted
test_predicted = pd.concat([test_input, test_output, test_predicted], axis=1)
#Guardamos el resultado en un csv
test_predicted.to_csv(output_csv, index=False)

    

