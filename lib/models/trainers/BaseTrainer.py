from lib.models import M2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
        pass

    @abstractmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        pass
    
    @staticmethod
    def fit_autokeras(model, X, y, designing, batch_size, callbacks = None, test_size:float=0.2):
        #Particionamos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #Entrenamos
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                            verbose=(1 if designing else 2), callbacks=callbacks, batch_size=batch_size)
        #Evaluamos
        score = model.evaluate(X_test, y_test, verbose=0)
        return model, score