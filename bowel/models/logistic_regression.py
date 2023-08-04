from bowel.models.model import Model
from bowel.utils.audio_utils import get_normalized_spectrogram, get_wav_length

from sklearn.linear_model import LogisticRegression
import yaml
import pickle
import os
import numpy as np


class LogisticRegression(Model):
    def __init__(self, config, input_dim=None, model_file=None):
        self.config = config
        if model_file is None:
            self.model = self.build_model()
        else:
            self.model = self.load(model_file)

    def build_model(self):
        return LogisticRegression(solver='lbfgs', max_iter=2000, verbose=1, n_jobs=-1)

    def load(self, model_file):
        return pickle.load(open(model_file, 'rb'))

    def train(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        return ''

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))
        with open(filename.replace(os.path.splitext(filename)[1], '.yml'), 'w') as config_file:
            yaml.dump(self.config, config_file, default_flow_style=False)

    def infer(self, wav_file):
        y_pred = []
        infer_batch_size = 1000
        duration = get_wav_length(wav_file)
        offset = 0.0
        i = 0
        while offset + self.config['chunk_length'] <= duration:
            X = []
            for _ in range(infer_batch_size):
                sample = np.array(get_normalized_spectrogram(wav_file, self.config, offset=offset, duration=self.config['wav_sample_length'], padding=True)).flatten()
                X.append(sample)
                i += 1
                offset = i * self.config['chunk_length'] / self.config['overlapping']
                if offset + self.config['wav_sample_length'] > duration:
                    break
            y_pred += self.model.predict(np.array(X)).tolist()
        frames = []
        for i, prediction in enumerate(y_pred):
            frames.append({'start': i*(self.config['chunk_length'] / self.config['overlapping']), 'end': i * (
                self.config['chunk_length'] / self.config['overlapping']) + self.config['chunk_length'], 'probability': prediction})
        return frames, duration
