from abc import ABC, abstractmethod

from bowel.models.crnn import CRNN
from bowel.models.rnn import RNN


class Semisupervised(ABC):
    def __init__(self, config, input_dim=None, model_file=None):
        self.config = config
        self.input_dim = input_dim
        self.model = self.build_model(model_file)

    def build_model(self, model_file=None):
        if self.config['model_type'] == 'convrnn':
            model = CRNN(self.config, self.input_dim, model_file)
        if self.config['model_type'] == 'convrnnnarrow':
            model = RNN(self.config, self.input_dim, model_file)
        return model
    
    def load(self, model_file):
        return self.model.load(model_file)

    @abstractmethod
    def train(self, X, y, X_unannotated, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)
    
    def summary(self):
        return self.model.summary()

    def save(self, filename):
        self.model.save(filename)

    def infer(self, wav_file):
        return self.model.infer(wav_file)
