import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path
import pickle
from lib.trainingcommon import load_real_track_data
from lib.trainingcommon import descale_pos_x
from lib.trainingcommon import descale_pos_y
from lib.filters.montecarlofilter import monte_carlo_filter

#Configuración
input_file_name = 'track_straight_01_all_sensors.mbd_v2_inputed_mediaprevpost'
model = 'cnn'
use_pos_z = False
scale_y = True
remove_not_full_rows = True

#Variables globales
track_file = './dataset_processed_csv/'+input_file_name+'.csv'
output_file = './prediction_output/'+model+'_'+input_file_name+'.csv'
model_dir = './models/'+model
scaler_file = model_dir+'/files/scaler.pkl'
model_file = model_dir+'/files/model.h5'
dim_x = 20.660138018121128
dim_y = 17.64103475472807



#Preparamos los datos
input_data, output_data = load_real_track_data(track_file, scaler_file, use_pos_z, scale_y, remove_not_full_rows)
print(input_data)

#Si el modelo es cnn, tenemos que darle una forma especial
if model == 'cnn':
  input_data = input_data.values.reshape(input_data.shape[0], input_data.shape[1], 1)

#Cargamos el modelo
model = tf.keras.models.load_model(model_file)

#Predecimos
predictions = model.predict(input_data)

print("-Predicciones-")
print("Real:")
print(output_data)
print("Estimado:")
print(predictions)

#Desescalamos
#with open(scaler_output_file, 'rb') as scalerFile:
#  scaler = pickle.load(scalerFile)
#  scalerFile.close()
#predictions = scaler.inverse_transform(predictions)

#Componemos la salida
output_data = output_data.to_numpy()
output_list = []
for index in range(0, len(predictions)):
  listrow = {
    'predicted_x': predictions[index][0],
    'predicted_y': predictions[index][1],
    #'predicted_z': predictions[index][2],
    'real_x': output_data[index][0],
    'real_y': output_data[index][1],
    #'real_z': output_data[index][2],
  }
  output_list.append(listrow)
output_data = pd.DataFrame(output_list)

#Si se ha escalado la salida, desescalamos
if scale_y:
  output_data['predicted_x'] = descale_pos_x(output_data['predicted_x'])
  output_data['predicted_y'] = descale_pos_y(output_data['predicted_y'])
  output_data['real_x'] = descale_pos_x(output_data['real_x'])
  output_data['real_y'] = descale_pos_y(output_data['real_y'])

#Preparamos cálculos
output_data['deviation_x'] = (output_data['predicted_x'] - output_data['real_x']).abs()
output_data['deviation_y'] = (output_data['predicted_y'] - output_data['real_y']).abs()
#output_data['deviation_z'] = (output_data['predicted_z'] - output_data['real_z']).abs()

#Filtramos
'''filtered_particles, filtered_predicted_data = monte_carlo_filter(output_data[['predicted_x', 'predicted_y']].to_numpy(), 100)

filtered_predicted_data = pd.DataFrame(filtered_predicted_data)
output_data['filtered_x'] = filtered_predicted_data[[0]]
output_data['filtered_y'] = filtered_predicted_data[[1]]
output_data['deviation_filtered_x'] = (output_data['filtered_x'] - output_data['real_x']).abs()
output_data['deviation_filtered_y'] = (output_data['filtered_y'] - output_data['real_y']).abs()'''



#Imprimimos la desviacion máxima minima y media de X e Y
print("- Desviaciones en predicciones -")
print("Desviación máxima X: "+str(output_data['deviation_x'].max()))
print("Desviación mínima X: "+str(output_data['deviation_x'].min()))
print("Desviación media X: "+str(output_data['deviation_x'].mean()))
print("Desviación máxima Y: "+str(output_data['deviation_y'].max()))
print("Desviación mínima Y: "+str(output_data['deviation_y'].min()))
print("Desviación media Y: "+str(output_data['deviation_y'].mean()))
'''print("- Desviaciones tras filtro -")
print("Desviación máxima X: "+str(output_data['deviation_filtered_x'].max()))
print("Desviación mínima X: "+str(output_data['deviation_filtered_x'].min()))
print("Desviación media X: "+str(output_data['deviation_filtered_x'].mean()))
print("Desviación máxima Y: "+str(output_data['deviation_filtered_y'].max()))
print("Desviación mínima Y: "+str(output_data['deviation_filtered_y'].min()))
print("Desviación media Y: "+str(output_data['deviation_filtered_y'].mean()))'''

#Hacemos la salida
output_data.to_csv(output_file, index=False)


#Mostramos el grafico
plt.plot([0, 0, dim_y, dim_y, 0], [0, dim_x,  dim_x, 0, 0], 'go-', label='Real', linewidth=1)
plt.plot(output_data['real_y'].values, output_data['real_x'].values, 'ro-', label='Real', linewidth=1)
plt.plot(output_data['predicted_y'].values, output_data['predicted_x'].values, 'mo-', label='Calculada', linewidth=1)
#plt.plot(output_data['filtered_y'].values, output_data['filtered_x'].values, 'bo-', label='Calculada', linewidth=1)
plt.show()