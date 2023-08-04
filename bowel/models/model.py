from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, config, model_file=None):
        self.config = config
        if model_file is None:
            self.model = self.build_model()
        else:
            self.model = self.load(model_file)

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def load(self, model_file):
        raise NotImplementedError

    @abstractmethod
    def train(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError
    
    @abstractmethod
    def summary(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, model_file):
        raise NotImplementedError
    
    @abstractmethod
    def infer(self, wav_file):
        raise NotImplementedError
