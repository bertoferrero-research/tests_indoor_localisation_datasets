import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import os.path

#Variables globales
training_file = './files/fingerprint_history_train.csv'
test_file = './files/fingerprint_history_test.csv'
model_file = './files/model.h5'

#Funciones
#Preparamos los datos para ser introducidos en el modelo
def prepare_data(data):
    #Extraemos cada parte
    y = data.iloc[:, 1:4]
    X = data.iloc[:, 4:]
    #Normalizamos los rssi a valores positivos de 0 a 1
    X += 100
    X /= 100
    #Convertimos a float32 para reducir complejidad
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    #Ordenamos alfabéticamente las columnas de X, asegurandonos de que todos los datasets van en el mismo orden
    X = X.reindex(sorted(X.columns), axis=1)
    #Devolvemos
    return X,y

def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')  
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()


#Cargamos los ficheros
train_data = pd.read_csv(training_file)
test_data = pd.read_csv(test_file)

#Preparamos los datos
X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

#Mostramos los valores de la primera columna
#pdTable = pd.DataFrame({'quantity acumulada':X_train.iloc(axis=1)[0]})
#pdTable.plot(kind='box')
#plt.show()

#Construimos el modelo
#Nos basamos en el diseño descrito en el paper "Indoor Localization using RSSI and Artificial Neural Network"
inputlength = X_train.shape[1]
outputlength = y_train.shape[1]
hiddenLayerLength = round(inputlength*2/3+outputlength, 0)
print("Tamaño de la entrada: "+str(inputlength))
print("Tamaño de la salida: "+str(outputlength))
print("Tamaño de la capa oculta: "+str(hiddenLayerLength))

input = tf.keras.layers.Input(shape=inputlength)
x = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
output = tf.keras.layers.Dense(outputlength, activation='relu')(x) #La salida son valores positivos
model = tf.keras.models.Model(inputs=input, outputs=output)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse','mae'] ) #mse y sgd sugeridos por chatgpt, TODO averiguar y entender por qué
#comparacion de optimizadores https://velascoluis.medium.com/optimizadores-en-redes-neuronales-profundas-un-enfoque-pr%C3%A1ctico-819b39a3eb5
print(model.summary())

#Entrenamos
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     batch_size=  32,
                     epochs=  25, 
                     verbose=1)

plot_learning_curves(history)

# Evaluamos usando el test set
score = model.evaluate(X_test, y_test, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

#Guardamos el modelo
if os.path.exists(model_file):
  os.remove(model_file)
model.save(model_file)