import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from lib.trainingcommon import load_data
from .ModelsBaseClass import ModelsBaseClass

class M4(ModelsBaseClass): 
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False,
                        scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, include_pos_z=use_pos_z, scale_y=scale_y, not_valid_sensor_value=100, return_valid_sensors_map=True)

    def build_model(self):
        pass

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        # Entradas
        inputSensors = ak.StructuredDataInput(name='input_sensors')
        InputMap = ak.StructuredDataInput(name='input_map')

        if designing:

            # Capas ocultas para cada entrada
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors')(inputSensors)
            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map')(InputMap)

            # Concatenamos las capas
            concat = ak.Merge()([hiddenLayer_sensors, hiddenLayer_map])

            # Capas ocultas tras la concatenación
            hiddenLayer = ak.DenseBlock(use_batchnorm=False)(concat)

        else:
            #Pendiente
            raise NotImplementedError
            # Capas ocultas para cada entrada
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors_1', num_layers=1, num_units=64)(inputSensors)
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors_2', num_layers=1, num_units=512)(hiddenLayer_sensors)

            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map', num_layers=1, num_units=1024)(InputMap)
            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map', num_layers=1, num_units=32)(hiddenLayer_map)
            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map', num_layers=1, num_units=128)(hiddenLayer_map)

            # Concatenamos las capas
            concat = ak.Merge(merge_type='concatenate')([hiddenLayer_sensors, hiddenLayer_map])

            # Capas ocultas tras la concatenación
            hiddenLayer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=256)(concat)
            hiddenLayer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=64)(hiddenLayer)

        # Salida
        output = ak.RegressionHead(metrics=['mse', 'accuracy'])(hiddenLayer)

        # Construimos el modelo
        model = ak.AutoModel(
                inputs=[inputSensors, InputMap],
                outputs=output, 
                overwrite=overwrite,
                tuner=tuner,
                seed=random_seed,
                max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder)

        return model