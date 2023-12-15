from lib.models import M6
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
from lib.trainingcommon import posXYlist_to_grid

class M6Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
               
        #Definimos el nombre del modelo y la configuración específica
        modelName = 'M6'
        cell_amount_x = 9
        cell_amount_y = 9

        #Cargamos los datos de entrenamiento
        X, y = M6.load_traning_data(dataset_path, scaler_file)

        y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)

        #Convertimos a categorical
        y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)
        
        #Instanciamos la clase del modelo
        modelInstance = M6(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_autokeras(model, X, y, designing, batch_size, callbacks=[callback])

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score