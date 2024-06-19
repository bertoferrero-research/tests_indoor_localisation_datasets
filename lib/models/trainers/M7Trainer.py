from sklearn.metrics import accuracy_score
from lib.models import M7
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
from lib.trainingcommon import posXYlist_to_grid, gridList_to_posXY
import autokeras as ak
from lib.trainingcommon import descale_numpy

class M7Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
               
        #Definimos el nombre del modelo y la configuración específica
        modelName = 'M7'
        cell_amount_x = 7
        cell_amount_y = 6

        #Cargamos los datos de entrenamiento
        X, y = M7.load_traning_data(dataset_path, scaler_file)

        y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)

        #Convertimos a categorical
        y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)
        
        #Instanciamos la clase del modelo
        modelInstance = M7(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_autokeras(model, X, y, designing, batch_size, callbacks=[callback])

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        cell_amount_x = 7
        cell_amount_y = 6

        #Cargamos los datos de entrenamiento
        input_data, output_data = M7.load_testing_data(dataset_path, scaler_file)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

        #Predecimos
        predictions = model.predict(input_data)
        predictions = np.argmax(predictions, axis=-1)
        # Convertimos a posiciones
        predictions_positions = gridList_to_posXY(
            predictions, cell_amount_x, cell_amount_y)
        
        # Evaluación
        output_data_grid = posXYlist_to_grid(output_data, cell_amount_x, cell_amount_y)
        output_data_categorical = tf.keras.utils.to_categorical(output_data_grid, num_classes=cell_amount_x*cell_amount_y)
        #accuracy = accuracy_score(output_data_grid, predictions)
        metrics = model.evaluate(input_data, output_data_categorical, verbose=0)
        formated_metrics = {
            'loss_mse': metrics[1],
            'accuracy': metrics[2]
        }

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions_positions, output_data, formated_metrics