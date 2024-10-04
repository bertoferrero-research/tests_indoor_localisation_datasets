from lib.models import M2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from abc import ABC, abstractmethod
import autokeras as ak

class BaseTrainer(ABC):
    @abstractmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
        pass

    @abstractmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        pass
  
    @abstractmethod
    def get_training_data(dataset_path: str, scaler_file: str):
        pass
    
    @staticmethod
    def get_model_instance(model_file: str):
        return tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)
    
    @staticmethod
    def fit_autokeras(model, X, y, designing, batch_size, callbacks = None, test_size:float=0.2):
        #Particionamos
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        #Entrenamos
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            verbose=(1 if designing else 2), callbacks=callbacks, batch_size=batch_size)
        #Evaluamos
        score = model.evaluate(X_val, y_val, verbose=0)
        return model, score