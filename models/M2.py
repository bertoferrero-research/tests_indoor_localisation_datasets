import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak

class M2:
    def __init__(self, inputlength, outputlength):
        self.inputlength = inputlength
        self.outputlength = outputlength

    def build_model(self):
        pass

    #Faltaría aquí un build_model_autokeras pero con M1 no es necesario
    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 1000):
        input = ak.StructuredDataInput()
        if desgining:
            hiddenLayers = ak.DenseBlock(use_batchnorm=False)(input)
        else:
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=128)(input)
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=128)(hiddenLayers)
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=1024)(hiddenLayers)
        
        output = ak.RegressionHead(output_dim=outputlength, metrics=['mse', 'accuracy'])(hiddenLayers)

        model = ak.AutoModel(
            inputs=input,
            outputs=output,
            overwrite=overwrite,
            seed=random_seed,
            max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder,
            tuner=tuner
        )

        return model