import os
import numpy as np

import yaml
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (Dense, Bidirectional, GRU, BatchNormalization, TimeDistributed, Dropout, Flatten)
from wandb.keras import WandbCallback

from bowel.models.model import Model
from bowel.utils.audio_utils import get_normalized_spectrogram, get_wav_length, split_spectrogram_to_windows


class RNN(Model):
    def __init__(self, config, input_dim=None, model_file=None):
        self.input_dim = input_dim
        super().__init__(config=config, model_file=model_file)

    def build_model(self):
        model = Sequential()
        model.add(TimeDistributed(Flatten(), input_shape=self.input_dim))
        model.add(TimeDistributed(Dense(120, activation='relu')))
        model.add(TimeDistributed(Dense(240, activation='relu')))
        model.add(TimeDistributed(Dense(360, activation='relu')))
        model.add(TimeDistributed(Dropout(0.4)))
        model.add(Bidirectional(GRU(80, activation='relu', return_sequences=True)))
        model.add(Bidirectional(GRU(80, activation='relu', return_sequences=True)))
        model.add(TimeDistributed(Dropout(0.4)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy', Precision(), Recall(), AUC(curve='PR')])
        return model
    
    def load(self, model_file):
        return load_model(model_file)

    def train(self, X, y, X_val=None, y_val=None):
        callbacks = [WandbCallback(), EarlyStopping(monitor='val_accuracy', mode='auto', patience=20, restore_best_weights=True)]
        if X_val is None:
            return self.model.fit(X, y, batch_size=self.config['batch_size'], epochs=self.config['epochs'], callbacks=callbacks, shuffle=True)
        return self.model.fit(X, y, batch_size=self.config['batch_size'], epochs=self.config['epochs'], validation_data=(X_val, y_val), callbacks=callbacks, shuffle=True)

    def predict(self, X):
        return self.model.predict(X)
    
    def summary(self):
        return self.model.summary()

    def save(self, filename):
        self.model.save(filename)
        with open(filename.replace(os.path.splitext(filename)[1], '.yml'), 'w') as config_file:
            yaml.dump(self.config, config_file, default_flow_style=False)

    def infer(self, wav_file):
        y_pred = []
        infer_batch_size = 100
        duration = get_wav_length(wav_file)
        step = min(self.config['wav_sample_length'], duration)
        offset = 0.0
        i = 0
        while offset + step <= duration:
            X = []
            for _ in range(infer_batch_size):
                norm_spec = get_normalized_spectrogram(wav_file, self.config, offset=offset, duration=step)
                sample = split_spectrogram_to_windows(norm_spec, step, self.config['overlapping'], self.config['chunk_length'])
                X.append(sample)
                i += 1
                offset = i * step
                if offset + step > duration:
                    break
            X = np.expand_dims(np.array(X), -1)
            y_pred += self.model.predict(X).flatten().tolist()
        frames = []
        for i, prediction in enumerate(y_pred):
            frames.append({'start': i*(self.config['chunk_length'] / self.config['overlapping']), 'end': i * (
                self.config['chunk_length'] / self.config['overlapping']) + self.config['chunk_length'], 'probability': prediction})
        return frames, duration
