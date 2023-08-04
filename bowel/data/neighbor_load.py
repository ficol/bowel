import os

import numpy as np
from multiprocessing import Pool

from bowel.utils.audio_utils import get_normalized_spectrogram, split_spectrogram_to_windows, get_wav_length, get_size
from bowel.data.load import Loader


class NeighborLoader(Loader):
    """A class to load audio data for simple models.
    """
    def __init__(self, data_dir, config, augmentation=False):
        super().__init__(data_dir, config)
        self.wav_files_y = dict()
        for file in self.wav_files:
            self.wav_files_y[file] = self.__get_categories(self.wav_files[file], get_wav_length(os.path.join(data_dir, file)))

    def __len__(self):
        """Gets size of dataset.

        Returns:
            int: Size of dataset.
        """
        return self.getlen()
    
    def getlen(self):
        size = 0
        for wav_file in self.wav_files:
            size += len(self.wav_files_y[wav_file])
        return size

    def __getitem__(self, index):
        """Gets sample from dataset.

        Args:
            index (int): Index of sample.

        Returns:
            tuple: Tuple of X and y values
        """
        return self.getitem(index)
    
    def getitem(self, index):
        file_index = index
        for file in self.wav_files:
            size = len(self.wav_files_y[file])
            if file_index < size:
                break
            file_index -= size
        audio_filename = os.path.join(self.data_dir, file)
        offset = file_index * self.config['chunk_length']
        X = np.array(get_normalized_spectrogram(audio_filename, self.config, offset, self.config['wav_sample_length'], True)).flatten()
        y = self.wav_files_y[file][file_index]
        return X, np.array(y)

    def get_data(self, processes=None):
        """Gets whole dataset.

        Returns:
            ndarray: Array of dataset.
        """
        pool = Pool(processes)
        data = pool.map(self.getitem, list(range(self.getlen())))
        X_data, y_data = zip(*data)
        return np.array(X_data), np.array(y_data)

    def __get_categories(self, df, length):
        categories = []
        start = 0.0
        end = start + self.config['chunk_length']
        i = 0
        for _, row in df.iterrows():
            while start < row['end'] and end <= length + 1e-8:
                if min(row['end'] - start, end - row['start']) > 0.5 * self.config['chunk_length']:
                    categories.append(1)
                else:
                    categories.append(0)
                i += 1
                start = i * self.config['chunk_length'] / self.config['overlapping']
                end = start + self.config['chunk_length']
        categories += int((length + 1e-8 - end) // (self.config['chunk_length'] / self.config['overlapping']) + 1) * [0]
        return categories
