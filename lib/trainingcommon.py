import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#region Carga de datos
def load_training_data(training_file: str, test_file: str, scaler_file: str=None, include_pos_z: bool=True, scale_y: bool=False, group_x_2dmap: bool=False):
    #Cargamos los ficheros
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    #Preparamos los datos
    X_train, y_train = prepare_data(train_data, include_pos_z, scale_y)
    X_test, y_test = prepare_data(test_data, include_pos_z, scale_y)

    #Escalamos
    if scaler_file is not None:
        scaler = StandardScaler()
        scaler.fit(X_train)
        with open(scaler_file, 'wb') as scalerFile:
            pickle.dump(scaler, scalerFile)
            scalerFile.close()

        X_train_scaled = X_train.copy()        
        X_train_scaled[X_train.columns] = scaler.transform(X_train)
        X_train = X_train_scaled
        X_test_scaled = X_test.copy()        
        X_test_scaled[X_test.columns] = scaler.transform(X_test)
        X_test = X_test_scaled

    #Agrupamos los valores de X en un mapa 2D
    if group_x_2dmap:
        X_train = group_rssi_2dmap(X_train)
        X_test = group_rssi_2dmap(X_test)

    #Devolvemos
    return X_train, y_train, X_test, y_test

def load_real_track_data(track_file: str, scaler_file: str=None, include_pos_z: bool=True, scale_y: bool=False):
    #Cargamos el fichero
    track_data = pd.read_csv(track_file)

    #Preparamos los datos
    X, y = prepare_data(track_data, include_pos_z, scale_y)

    #Escalamos
    if scaler_file is not None:
        with open(scaler_file, 'rb') as scalerFile:
            scaler = pickle.load(scalerFile)
            scalerFile.close()

        X_scaled = X.copy()        
        X_scaled[X.columns] = scaler.transform(X)
        X = X_scaled   

    #Devolvemos
    return X, y

def prepare_data(data, include_pos_z: bool=True, scale_y: bool=False):
    #Extraemos cada parte
    y = data.iloc[:, 1:(4 if include_pos_z else 3)]
    X = data.iloc[:, 4:]
    #Escalamos y
    if scale_y:
        y = scale_dataframe(y)
    #Convertimos a float32 e in32 para reducir complejidad
    y = y.astype(np.float32)
    X = X.astype(np.int32)
    #Por cada columna de X añadimos otra indicando si ese nodo ha de tenerse o no en cuenta
    #nodes = X.columns
    #for node in nodes:
    #  X[node+"_on"] = (X[node] > 0).astype(np.int32)

    #Ordenamos alfabéticamente las columnas de X, asegurandonos de que todos los datasets van en el mismo orden
    X = X.reindex(sorted(X.columns), axis=1)
    #Devolvemos
    return X,y


def group_rssi_2dmap(data: pd.DataFrame, default_empty_value: int=-200):
    #Definimos el array de la matriz a extrapolar, sacada de los mapas del dataset
    #  0  12  21  0
    #  11 10  20  22
    #  42 40  30  31
    #  0  41  32  0
    rssi_map = [
        [None, '000000000102', '000000000201', None],
        ['000000000101', 'b827eb4521b4', 'b827eb917e19', '000000000202'],
        ['000000000402', 'b827ebfd7811', 'b827ebf7d096', '000000000301'],
        [None, '000000000401', '000000000302', None]
    ]
    final_data = []
    #Por cada fila del dataset creamos un array con los valores de rssi en el mismo índice que la matriz rssi_map
    for index, row in data.iterrows():
        rssi = np.ndarray((len(rssi_map), len(rssi_map[0])))
        for i in range(len(rssi_map)):
            rssi_map_row = rssi_map[i]
            for j in range(len(rssi_map_row)):
                rssi_map_col = rssi_map_row[j]
                rssi_value = default_empty_value
                if rssi_map_col is not None:
                    rssi_value = row[rssi_map_col]
                rssi[i][j] = rssi_value
        final_data.append(rssi)
    #Devolvemos
    return np.array(final_data)
#endregion

#region escalado de datos

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


#region dibujado
def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')  
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()
#endregion