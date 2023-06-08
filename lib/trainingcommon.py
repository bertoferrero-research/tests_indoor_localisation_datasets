import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#region Carga de datos
def load_training_data(training_file: str, test_file: str, scaler_file: str=None, include_pos_z: bool=True, scale_y: bool=False):
    #Cargamos los ficheros
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    #Preparamos los datos
    X_train, y_train = prepare_training_data(train_data, include_pos_z, scale_y)
    X_test, y_test = prepare_training_data(test_data, include_pos_z, scale_y)

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

def prepare_training_data(data, include_pos_z: bool=True, scale_y: bool=False):
    #Extraemos cada parte
    y = data.iloc[:, 1:(4 if include_pos_z else 3)]
    X = data.iloc[:, 4:]
    #Escalamos y
    if scale_y:
        y = scale_dataframe(y)
    #Convertimos a float32 e in32 para reducir complejidad
    y = y.astype(np.float32)
    X = X.astype(np.int32)

    #Ordenamos alfabéticamente las columnas de X, asegurandonos de que todos los datasets van en el mismo orden
    X = X.reindex(sorted(X.columns), axis=1)
    #Devolvemos
    return X,y
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
    data['pos_x'] = scale_pos_x(data['pos_x'])
    data['pos_y'] = scale_pos_y(data['pos_y'])
    return data

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
    data['pos_x'] = descale_pos_x(data['pos_x'])
    data['pos_y'] = descale_pos_y(data['pos_y'])
    return data

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