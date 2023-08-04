import os
import pandas as pd

from abc import ABC, abstractmethod


class Loader(ABC):
    """A class to load audio data.
    """

    def __init__(self, data_dir, config):
        """Loader constructor.

        Args:
            data_dir (str): Path to directory with processed audio data.
            config (dict): Dictionary with config parameters.
        """
        self.data_dir = data_dir
        self.unannotated_dir = os.path.join(self.data_dir, 'unannotated')
        self.config = config
        self.wav_files = {file: pd.read_csv(os.path.join(data_dir, file.replace('.wav', '.csv'))) for file in os.listdir(data_dir) if file.endswith('.wav')}
        self.unannotated_wav_files = []
        if os.path.isdir(os.path.join(data_dir, 'unannotated')):
            self.unannotated_wav_files = [file for file in os.listdir(os.path.join(data_dir, 'unannotated'))]

    @abstractmethod
    def __len__(self):
        """Gets size of dataset.

        Returns:
            int: Size of dataset.
        """
        return NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """Gets sample from dataset.

        Args:
            index (int): Index of sample.

        Returns:
            tuple: Tuple of X and y values
        """
        return NotImplementedError
