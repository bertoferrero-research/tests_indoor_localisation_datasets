class ModelsBaseClass:

    def __init__(self, inputlength, outputlength):
        self.inputlength = inputlength
        self.outputlength = outputlength

    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        raise NotImplementedError

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        raise NotImplementedError
        
    def build_model(self):
        raise NotImplementedError

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        raise NotImplementedError
        