from lib.models import M4
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
import autokeras as ak
from lib.trainingcommon import descale_numpy

class M4Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
               
        #Definimos el nombre del modelo
        modelName = 'M4'

        #Cargamos los datos de entrenamiento
        X, y, Xmap = M4.load_traning_data(dataset_path, scaler_file)

        #Convertimos a numpy y formato
        X = X.to_numpy()
        y = y.to_numpy()
        Xmap = Xmap.to_numpy()

        #Instanciamos la clase del modelo
        modelInstance = M4(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)

        # Entrenamos
        X_train, X_test, y_train, y_test, Xmap_train, Xmap_test = train_test_split(
            X, y, Xmap, test_size=0.2)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        history = model.fit([X_train, Xmap_train], y_train, validation_data=([X_test, Xmap_test], y_test),
                            verbose=(1 if designing else 2), callbacks=[callback], batch_size=batch_size)

        # Evaluamos usando el test set
        score = model.evaluate([X_test, Xmap_test], y_test, verbose=0)

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        #Cargamos los datos de entrenamiento
        input_data, output_data, input_map_data = M4.load_testing_data(dataset_path, scaler_file)

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

        #Predecimos
        predictions = model.predict([input_data, input_map_data])

        #Los datos de predicción y salida vienen escalados, debemos desescalarlos
        output_data = output_data.to_numpy()
        output_data = descale_numpy(output_data)
        predictions = descale_numpy(predictions)

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions, output_data