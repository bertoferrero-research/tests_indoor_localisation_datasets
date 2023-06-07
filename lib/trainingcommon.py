import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def load_training_data(training_file: str, test_file: str, scaler_file: str=None):
    #Cargamos los ficheros
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    #Preparamos los datos
    X_train, y_train = prepare_training_data(train_data)
    X_test, y_test = prepare_training_data(test_data)

    #Escalamos
    if scaler_file is not None:
        scaler = StandardScaler()
        scaler.fit(X_train)
        with open(scaler_file, 'wb') as scalerFile:
            pickle.dump(scaler, scalerFile)
            scalerFile.close()

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    #Devolvemos
    return X_train, y_train, X_test, y_test

def prepare_training_data(data):
    #Extraemos cada parte
    y = data.iloc[:, 1:4]
    X = data.iloc[:, 4:]
    #Normalizamos los rssi a valores positivos de 0 a 1
    #X += 100
    #X /= 100
    #Convertimos a float32 para reducir complejidad
    X = X.astype(np.int32)
    y = y.round(4).astype(np.float32)
    #Por cada columna de X añadimos otra indicando si ese nodo ha de tenerse o no en cuenta
    #nodes = X.columns
    #for node in nodes:
    #  X[node+"_on"] = (X[node] > 0).astype(np.int32)

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