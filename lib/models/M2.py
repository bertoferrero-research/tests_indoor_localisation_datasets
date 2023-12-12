import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from lib.trainingcommon import load_data
from .ModelsBaseClass import ModelsBaseClass

class M2(ModelsBaseClass): 
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=True)

    def build_model(self):
        pass

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        input = ak.StructuredDataInput()
        if designing:
            hiddenLayers = ak.DenseBlock(use_batchnorm=False)(input)
        else:
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=128)(input)
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=128)(hiddenLayers)
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=1024)(hiddenLayers)
        
        output = ak.RegressionHead(output_dim=self.outputlength, metrics=['mse', 'accuracy'])(hiddenLayers)

        model = ak.AutoModel(
            inputs=input,
            outputs=output,
            overwrite=overwrite,
            seed=random_seed,
            max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder,
            tuner=tuner
        )

        return model