from lib.models import M8
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
from lib.trainingcommon import posXYlist_to_grid, gridList_to_posXY
import keras_tuner
from lib.trainingcommon import descale_numpy

class M8Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
               
        #Definimos el nombre del modelo y la configuración específica
        modelName = 'M8'
        cell_amount_x = 3
        cell_amount_y = 3
        max_trials = 100

        #Cargamos los datos de entrenamiento
        X, y = M8.load_traning_data(dataset_path, scaler_file)

        #Cargamos cada dimension
        y_dim1 = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)
        y_dim2 = posXYlist_to_grid(y.to_numpy(), cell_amount_x**2, cell_amount_y**2)

        #Convertimos a categorical
        y_dim1 = tf.keras.utils.to_categorical(y_dim1, num_classes=cell_amount_x*cell_amount_y)
        y_dim2 = tf.keras.utils.to_categorical(y_dim2, num_classes=(cell_amount_x**2)*(cell_amount_y**2))
        
        #Instanciamos la clase del modelo
        inputlength = X.shape[1]
        outputlength_dim1 = y_dim1.shape[1]
        outputlength_dim2 = y_dim2.shape[1]
        modelInstance = M8(inputlength, [outputlength_dim1, outputlength_dim2])

        #Creamos la función hypermodelo para el entrenamiento
        def build_model(hp):
            #Definimos los hipermodelos y construimos el modelo en si
            loss = 'categorical_crossentropy'
            activation = 'softmax'
            inputdropout = hp.Float('input_dropout', min_value=0.0, max_value=0.5, step=0.1)
            dimension1 = []
            dimension2 = []
            if designing:
                for i in range(hp.Int('num_layers_dimension1', min_value=1, max_value=10, step=1)):
                    dimension1.append(
                        {
                            'units': hp.Int('units_dimension1_'+str(i), min_value=4, max_value=2048, step=2, sampling="log"), 
                            'dropout': hp.Float('dropout_dimension1_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        })

                for i in range(hp.Int('num_layers_dimension2', min_value=1, max_value=10, step=1)):
                    dimension2.append(
                        {
                            'units': hp.Int('units_dimension2_'+str(i), min_value=4, max_value=2048, step=2, sampling="log"),
                            'dropout': hp.Float('dropout_dimension2_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        }
                    )
                loss_weight_d1 = hp.Float('loss_weight_d1', min_value=0.1, max_value=0.9, step=0.1)
                learning_rate = learning_rate=hp.Float('learning_rate', min_value=0.00001, max_value=0.1, step=10, sampling="log")

            else:
                d1units = [1024, 128, 2048, 1024, 16, 16, 1024, 4]
                for i in range(len(d1units)):
                    dimension1.append(
                        {
                            'units': d1units[i], 
                            'dropout': hp.Float('dropout_dimension1_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        })
                
                d2units = [128, 8, 8]
                for i in range(len(d2units)):
                    dimension2.append(
                        {
                            'units': d2units[i], 
                            'dropout': hp.Float('dropout_dimension2_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        })
                loss_weight_d1 = 0.5
                learning_rate = 0.01            

            loss_weight_d2 = 1 - loss_weight_d1
            
            return modelInstance.build_custom_model(dimension1=dimension1, dimension2=dimension2, loss_weight_d1=loss_weight_d1, loss_weight_d2=loss_weight_d2, inputdropout=inputdropout, learning_rate=learning_rate, loss=loss, activation_d1=activation, activation_d2=activation)
            

        #Entrenamos
        callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_output_d2_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback3 = tf.keras.callbacks.EarlyStopping(monitor='val_output_d1_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        X_train, X_test, y_dim1_train, y_dim1_test, y_dim2_train, y_dim2_test = train_test_split(X, y_dim1, y_dim2, test_size=0.2)
        keras_tuner.GridSearch
        tuner = keras_tuner.BayesianOptimization(
            build_model,
            objective=[keras_tuner.Objective("val_output_d2_accuracy", direction="max"), keras_tuner.Objective("val_output_d1_accuracy", direction="max"), keras_tuner.Objective("val_loss", direction="min")],
            max_trials=max_trials,
            overwrite=overwrite,
            directory=tmp_dir,
            project_name=modelName,
            seed=random_seed
        )

        tuner.search(X_train, [y_dim1_train, y_dim2_train], epochs=1000, validation_data=(X_test, [y_dim1_test, y_dim2_test]), 
                        verbose=2,
                        batch_size=batch_size,
                        callbacks=[callback1, callback2, callback3])

        # Devolvemos el modelo entrenado
        model = tuner.get_best_models()[0]
        model.build(input_shape=(inputlength,))
        score = model.evaluate(X_test, [y_dim1_test, y_dim2_test], verbose=0)

        # Imprimimos la mejor configuración
        best_hps = tuner.get_best_hyperparameters()[0]
        print(best_hps.values)

        return model, score

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        cell_amount_x = 3
        cell_amount_y = 3

        #Cargamos los datos de entrenamiento
        input_data, output_data = M8.load_testing_data(dataset_path, scaler_file)

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file)

        #Predecimos
        predictions = model.predict(input_data)
        #Nos quedamos con la salida final
        predictions = predictions[-1]
        #Usamp argmax para quedarnos con la probabilidad más alta de cada posición
        predictions = np.argmax(predictions, axis=-1)
        # Convertimos a posiciones
        predictions_positions = gridList_to_posXY(predictions, cell_amount_x=cell_amount_x**2, cell_amount_y=cell_amount_y**2)

        #Devolvemos las predicciones y los datos de salida esperados
        output_data = output_data.to_numpy()
        return predictions_positions, output_data