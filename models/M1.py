import numpy as np
import pandas as pd
import tensorflow as tf

class M1:
    def __init__(self, inputlength, outputlength):
        self.inputlength = inputlength
        self.outputlength = outputlength

    def build_model(self):
        input = tf.keras.layers.Input(shape=self.inputlength)        
        hiddenLayerLength = round(self.inputlength*2/3+self.outputlength, 0)
        hiddenLayer = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
        output = tf.keras.layers.Dense(self.outputlength, activation='linear')(hiddenLayer)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'] )

        return model

    #Faltaría aquí un build_model_autokeras pero con M1 no es necesario
    def build_model_autokeras(self):
        pass