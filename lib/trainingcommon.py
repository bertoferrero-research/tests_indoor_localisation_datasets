import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#region Carga de datos
def load_training_data(training_file: str, scaler_file: str=None, include_pos_z: bool=True, scale_y: bool=False, remove_not_full_rows: bool=False):
    #Cargamos los ficheros
    train_data = pd.read_csv(training_file)

    #Preparamos los datos
    X, y = prepare_data(train_data, include_pos_z, scale_y, remove_not_full_rows)

    #Escalamos
    if scaler_file is not None:
        X = scale_RSSI_training(scaler_file, X)

    #Devolvemos
    return X, y

def load_training_data_inverse(training_file: str, scaler_file: str, include_pos_z: bool=True, scale_y: bool=False, remove_not_full_rows: bool=False, separate_mac_and_pos: bool=False):
    '''
    Devuelve los datos de entrenamiento y test preparados para entrenar la predicción de posicion a rssi.
    La salida para X contendrá tres columnas, pos_x, pos_y y mac
    La salida para y contendrá el valor rssi correspondiente
    '''
    #Recogemos los valores originales
    X, y = load_training_data(training_file, test_file, None, include_pos_z, scale_y, remove_not_full_rows)

    #Acumulamos los datos en el formato nuevo
    sensors = X.columns.to_list()
    sensors.sort()
    train_data = []
    for index, row in y.iterrows():
        for i in range(len(sensors)):
            sensor = sensors[i]
            train_data.append({
                'pos_x': row['pos_x'],
                'pos_y': row['pos_y'],
                'sensor_mac': i,
                'rssi': X[sensor][index]
            })
    train_data = pd.DataFrame(train_data)

    #Convertimos los dtype
    train_data['pos_x'] = pd.to_numeric(train_data['pos_x'], downcast='float')
    train_data['pos_y'] = pd.to_numeric(train_data['pos_y'], downcast='float')
    train_data['sensor_mac'] = pd.to_numeric(train_data['sensor_mac'], downcast='integer')
    train_data['rssi'] = pd.to_numeric(train_data['rssi'], downcast='float')

    #Dividimos X e y
    X = train_data[['pos_x', 'pos_y', 'sensor_mac']]
    y = train_data[['rssi']]

    #Escalamos el rssi
    y = scale_RSSI_training(scaler_file, y)

    #Separamos mac y pos
    if separate_mac_and_pos:
        X = [X[['pos_x', 'pos_y']], X[['sensor_mac']]]
    
    #Devolvemos
    return X, y, sensors

def load_real_track_data(track_file: str, scaler_file: str=None, include_pos_z: bool=True, scale_y: bool=False, remove_not_full_rows: bool=False):
    #Cargamos el fichero
    track_data = pd.read_csv(track_file)

    #Preparamos los datos
    X, y = prepare_data(track_data, include_pos_z, scale_y, remove_not_full_rows)

    #Escalamos
    if scaler_file is not None:
        X = scale_RSSI_track(scaler_file, X)

    #Devolvemos
    return X, y

def load_real_track_data_inverse(track_file: str, scaler_file: str, include_pos_z: bool=True, scale_y: bool=False, remove_not_full_rows: bool=False, separate_mac_and_pos: bool=False):
    '''
    Devuelve los datos de entrenamiento y test preparados para entrenar la predicción de posicion a rssi.
    La salida para X contendrá tres columnas, pos_x, pos_y y mac
    La salida para y contendrá el valor rssi correspondiente
    '''
    #Recogemos los valores originales
    X, y = load_real_track_data(track_file, None, include_pos_z, scale_y, remove_not_full_rows)

    #Acumulamos los datos en el formato nuevo
    sensors = X.columns.to_list()
    sensors.sort()
    data = []
    for index, row in y.iterrows():
        for i in range(len(sensors)):
            sensor = sensors[i]
            data.append({
                'pos_x': row['pos_x'],
                'pos_y': row['pos_y'],
                'sensor_mac': i,
                'rssi': X[sensor][index]
            })
    data = pd.DataFrame(data)

    #Convertimos los dtype
    data['pos_x'] = pd.to_numeric(data['pos_x'], downcast='float')
    data['pos_y'] = pd.to_numeric(data['pos_y'], downcast='float')
    data['sensor_mac'] = pd.to_numeric(data['sensor_mac'], downcast='integer')
    data['rssi'] = pd.to_numeric(data['rssi'], downcast='float')

    X = data[['pos_x', 'pos_y', 'sensor_mac']]
    y = data[['rssi']]

    #Escalamos el rssi
    y = scale_RSSI_track(scaler_file, y)

    if separate_mac_and_pos:
        X = [X[['pos_x', 'pos_y']], X[['sensor_mac']]]
    
    #Devolvemos
    return X, y, sensors

def prepare_data(data, include_pos_z: bool=True, scale_y: bool=False, remove_not_full_rows: bool=False):
    #Eliminamos las filas que no tienen todos los datos
    
    if remove_not_full_rows:
        #Reemplazamos los -200 en las columnas de rssi (a partir de la cuarta) por NaN
        data.iloc[:, 4:] = data.iloc[:, 4:].replace(-200, np.nan)

    #Extraemos cada parte
    y = data.iloc[:, 1:(4 if include_pos_z else 3)]
    X = data.iloc[:, 4:]
    #Escalamos y
    if scale_y:
        y = scale_dataframe(y)

    #Ordenamos alfabéticamente las columnas de X, asegurandonos de que todos los datasets van en el mismo orden
    X = X.reindex(sorted(X.columns), axis=1)

    #Inputamos los valores de -200 por el modelo
    #YA no es necesario, hay un csv con los datos inputados
    #if remove_not_full_rows:
    #    X = imputing_predict_na_data(X)

    #Convertimos a float32 e in32 para reducir complejidad
    #y = y.astype(np.float32)
    #X = X.astype(np.int32)

    #Devolvemos
    return X,y

def imputing_predict_na_data(data: pd.DataFrame):
    #Creamos una copia de data para el trabajo
    data_tmp = data.copy()
    #Reemplazamos los -200 por NaN
    data = data.replace(-200, np.nan)
    #Borramos las filas que tengan 3 o mas NaN
    #data_tmp = data_tmp.dropna(thresh=3)
    #Nos aseguramos que en el tmp sea -200
    data_tmp = data_tmp.replace(np.nan, -200)

    #Cargamos el scaler
    scaler = MinMaxScaler()
    scaler.fit([[-100], [0]])
    #Escalamos
    for sensor in data_tmp.columns:
        data_tmp[sensor] = scaler.transform(data_tmp[sensor].values.reshape(-1, 1)).flatten()

    #Cargamos el modelo
    model = tf.keras.models.load_model('dataset_imputing_values/files/model.h5')
    #Predecimos
    data_tmp_output = model.predict(data_tmp.values.reshape(data_tmp.shape[0], data_tmp.shape[1], 1))
    #Desescalamos
    data_tmp_output = scaler.inverse_transform(data_tmp_output.reshape(data_tmp_output.shape[0], data_tmp_output.shape[1]))
    data_tmp_output = pd.DataFrame(data_tmp_output, columns=data_tmp.columns).round()

    #Reemplazamos los -200 por los valores predichos
    for index,row in data.iterrows():
        for sensor in data.columns:
            if np.isnan(row[sensor]):
                data[sensor][index] = data_tmp_output[sensor][index]

    #Devolvemos
    return data


#endregion

#region escalado de datos

def scale_RSSI_training(scaler_file: str, X_data):
    scaler = StandardScaler()
    scaler.fit(X_data)
    with open(scaler_file, 'wb') as scalerFile:
        pickle.dump(scaler, scalerFile)
        scalerFile.close()

    X_data_scaled = X_data.copy()        
    X_data_scaled[X_data.columns] = scaler.transform(X_data)
    X_data = X_data_scaled

    return X_data

def scale_RSSI_track(scaler_file: str, X):
    with open(scaler_file, 'rb') as scalerFile:
        scaler = pickle.load(scalerFile)
        scalerFile.close()

    X_scaled = X.copy()        
    X_scaled[X.columns] = scaler.transform(X)
    X = X_scaled   

    return X

def get_scaler_pos_x():
    """
    Devuelve un scaler para la posición x
    Returns: 
        MinMaxScaler    
    """
    scaler = MinMaxScaler()
    scaler.fit([[0],[20.660138018121128]])
    return scaler

def get_scaler_pos_y():
    """
    Devuelve un scaler para la posición y
    Returns: 
        MinMaxScaler    
    """
    scaler = MinMaxScaler()
    scaler.fit([[0],[17.64103475472807]])
    return scaler

def scale_pos_x(pos_x: pd.Series):
    """
    Escala la posición x
    Args:
        pos_x (pd.Series): posición x
    Returns:
        pd.Series: posición x escalada
    """
    scaler = get_scaler_pos_x()
    return scaler.transform(pos_x.values.reshape(-1, 1)).flatten()

def scale_pos_y(pos_y: pd.Series):
    """
    Escala la posición y
    Args:
        pos_y (pd.Series): posición y
    Returns:
        pd.Series: posición y escalada
    """
    scaler = get_scaler_pos_y()
    return scaler.transform(pos_y.values.reshape(-1, 1)).flatten()

def scale_dataframe(data: pd.DataFrame):
    """
    Escala un dataframe
    Args:
        data (pd.DataFrame): dataframe a escalar
    Returns:
        pd.DataFrame: dataframe escalado
    """
    data_scaled = data.copy()
    data_scaled['pos_x'] = scale_pos_x(data['pos_x'])
    data_scaled['pos_y'] = scale_pos_y(data['pos_y'])
    return data_scaled

def descale_pos_x(pos_x: pd.Series):
    """
    Desescala la posición x
    Args:
        pos_x (pd.Series): posición x
    Returns:
        pd.Series: posición x desescalada
    """
    scaler = get_scaler_pos_x()
    return scaler.inverse_transform(pos_x.values.reshape(-1, 1)).flatten()

def descale_pos_y(pos_y: pd.Series):
    """
    Desescala la posición y
    Args:
        pos_y (pd.Series): posición y
    Returns:
        pd.Series: posición y desescalada
    """
    scaler = get_scaler_pos_y()
    return scaler.inverse_transform(pos_y.values.reshape(-1, 1)).flatten()

def descale_dataframe(data: pd.DataFrame):
    """
    Desescala un dataframe
    Args:
        data (pd.DataFrame): dataframe a desescalar
    Returns:
        pd.DataFrame: dataframe desescalado
    """
    data_scaled = data.copy()
    data_scaled['pos_x'] = descale_pos_x(data['pos_x'])
    data_scaled['pos_y'] = descale_pos_y(data['pos_y'])
    return data_scaled

#endregion


#region Analisis y dibujado

def cross_val_score_multi_input(model:tf.keras.Model, X, y, cv, loss, optimizer, metrics, batch_size:int, epochs:int, verbose = 0):
    '''
    Función que realiza cross validation de un modelo con múltiples entradas
    Args:
        model (keras.model): modelo a entrenar
        X (list): lista de arrays de entrada
        y (array): array de salida
        cv (sklearn.model_selection.KFold): objeto de validación cruzada
        batch_size (int): tamaño del batch
        epochs (int): número de épocas
        verbose (int, optional): nivel de verbosidad. Defaults to 0.
    Returns:
        np.array: lista de scores
    '''
    #Basado en https://stackoverflow.com/questions/59350224/crossvalidation-of-keras-model-with-multiply-inputs-with-scikit-learn
    cv_score = []
    for i, (train, test) in enumerate(cv.split(X[0],y)): #Las dimensiones deben ser siempre las mismas. seguimos el ejemplo y usamso el primer índice
        #Clonamos el modelo para resetear los pesos
        model_clone = tf.keras.models.clone_model(model)
        model_clone.compile(loss=loss, optimizer=optimizer, metrics=[metrics] )

        #Preparamos la entrada X para train y test
        X_train = []
        X_test = []
        for j in range(len(X)):
            if type(X[j]) == pd.DataFrame:
                X_train.append(X[j].iloc[train])
                X_test.append(X[j].iloc[test])
            else:                
                X_train.append(X[j][train])
                X_test.append(X[j][test])
        
        #Ahora las Y
        if type(y) == pd.DataFrame:
            y_train = y.iloc[train]
            y_test = y.iloc[test]
        else:
            y_train = y[train]
            y_test = y[test]

        #Entrenamos el modelo y acumulamos el score
        print("Running Fold", i+1)
        model_clone.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        result = model_clone.evaluate(X_test, y_test, verbose=verbose)
        cv_score.append(result[0]) #0 es el loss siempre, si hay más de una métrica, se incrementa el índice
        tf.keras.backend.clear_session()
    return np.array(cv_score)

def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')  
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()
#endregion