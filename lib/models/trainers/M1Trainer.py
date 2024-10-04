from lib.models import M1
from lib.trainingcommon import descale_numpy
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import autokeras as ak

class M1Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
               
        #Definimos el nombre del modelo
        modelName = 'M1'

        #Cargamos los datos de entrenamiento
        X, y = M1.load_traning_data(dataset_path, scaler_file)

        #Instanciamos la clase del modelo
        modelInstance = M1(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_autokeras(model, X, y, designing, batch_size, callbacks=[callback])

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score
    
    @staticmethod
    def get_training_data(dataset_path: str, scaler_file: str):
        input_data, output_data = M1.load_testing_data(dataset_path, scaler_file)
        return input_data, output_data


    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        #Cargamos los datos de entrenamiento
        input_data, output_data = M1Trainer.get_training_data(dataset_path, scaler_file)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = BaseTrainer.get_model_instance(model_file)

        #Evaluamos
        metrics = model.evaluate(input_data, output_data, verbose=0)

        #Predecimos
        predictions = model.predict(input_data)

        #Los datos de predicción y salida vienen escalados, debemos desescalarlos
        output_data = descale_numpy(output_data)
        predictions = descale_numpy(predictions)

        #Formateamos las métricas
        formated_metrics = {
            'loss_mse': metrics[1],
            'accuracy': metrics[2]
        }

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions, output_data, formated_metrics

