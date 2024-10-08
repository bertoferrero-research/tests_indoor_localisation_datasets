import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from lib.trainingcommon import load_data
from .ModelsBaseClass import ModelsBaseClass

class M1(ModelsBaseClass):

    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=True)

    def build_model(self):
        input = tf.keras.layers.Input(shape=self.inputlength)        
        hiddenLayerLength = round(self.inputlength*2/3+self.outputlength, 0)
        hiddenLayer = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
        output = tf.keras.layers.Dense(self.outputlength, activation='linear')(hiddenLayer)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'] )

        return model

    #Faltaría aquí un build_model_autokeras pero con M1 no es necesario
    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        input = ak.StructuredDataInput()
        hiddenLayerLength = round(self.inputlength*2/3+self.outputlength, 0)
        hiddenLayers = ak.DenseBlock(num_layers=1, num_units=hiddenLayerLength, use_batchnorm=False)(input)
        output = ak.RegressionHead(metrics=['mse', 'accuracy'])(hiddenLayers)

        model = ak.AutoModel(
            inputs=input,
            outputs=output,
            overwrite=overwrite,
            seed=random_seed,
            max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder,
            tuner=tuner
        )

        return model