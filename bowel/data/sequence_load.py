import os

import numpy as np
from multiprocessing import Pool

from bowel.utils.audio_utils import (get_normalized_spectrogram, split_spectrogram_to_windows, get_wav_length, get_size, flip, 
                                        freq_mask, time_mask, mixup, gaussian, shuffle_events, shuffle_intervals, is_true)
from bowel.data.load import Loader


class SequenceLoader(Loader):
    """A class to load audio data for sequence models.
    """

    def __init__(self, data_dir, config, augmentation=False):
        self.augmentation = augmentation
        self.sample_overlapping = 1.0
        if self.augmentation:
            self.sample_overlapping = config['augmentation_overlapping']
        super().__init__(data_dir, config)

    def __len__(self):
        """Gets size of dataset.

        Returns:
            int: Size of dataset.
        """
        return self.getlen(augmented=True)
    
    def getlen(self, augmented=False):
        size = 0
        for wav_file in self.wav_files:
            size += get_size(get_wav_length(os.path.join(self.data_dir, wav_file)),
                             self.config['wav_sample_length'], self.sample_overlapping)
        if augmented:
            size += int(self.config['augmentation_size'] * size)
        return size

    def getlen_unannotated(self):
        size = 0
        for wav_file in self.unannotated_wav_files:
            size += get_size(get_wav_length(os.path.join(self.unannotated_dir, wav_file)),
                             self.config['wav_sample_length'], 1.0)
        return size

    def __getitem__(self, index):
        """Gets sample from dataset.

        Args:
            index (int): Index of sample.

        Returns:
            tuple: Tuple of X and y values
        """
        return self.getitem(index)
    
    def getitem(self, index, flipped=False, freq_masked=False, noise_gauss=False, permutation=False, shuffle=False, time_masked=False):
        file_index = index % self.getlen()
        for file in self.wav_files:
            size = get_size(get_wav_length(os.path.join(self.data_dir, file)),
                             self.config['wav_sample_length'], self.sample_overlapping)
            if file_index < size:
                break
            file_index -= size
        audio_filename = os.path.join(self.data_dir, file)
        offset = file_index * \
            (self.config['wav_sample_length'] / self.sample_overlapping)
        norm_spec = get_normalized_spectrogram(
            audio_filename, self.config, offset, self.config['wav_sample_length'])
        y = self.__get_categories(self.wav_files[file], file_index)
        if flipped:
            norm_spec, y = flip(norm_spec, y)
        if freq_masked:
            norm_spec = freq_mask(norm_spec, self.config['augmentation_freq_masking'])
        if time_masked:
            norm_spec = time_mask(norm_spec, self.config['augmentation_time_masking'])
        if noise_gauss:
            norm_spec = gaussian(norm_spec, self.config['augmentation_gaussian_std'])
        if permutation:
            multiply_step = int(self.config['chunk_length']  / self.config['hop_length'])
            norm_spec, y = shuffle_events(norm_spec, [i for i in y for _ in range(multiply_step)])
            y = y[::multiply_step]
        if shuffle:
            multiply_step = int(self.config['chunk_length']  / self.config['hop_length'])
            norm_spec, y = shuffle_intervals(norm_spec, [i for i in y for _ in range(multiply_step)], self.config['augmentation_shuffle_intervals'])
            y = y[::multiply_step]
        X = split_spectrogram_to_windows(
            norm_spec, self.config['wav_sample_length'], self.config['overlapping'], self.config['chunk_length'])
        return np.expand_dims(np.array(X), -1), np.array(y)
    
    def getitem_unannotated(self, index, flipped=False, freq_masked=False, noise_gauss=False, permutation=False, shuffle=False, time_masked=False):
        for file in self.unannotated_wav_files:
            size = get_size(get_wav_length(os.path.join(self.unannotated_dir, file)),
                             self.config['wav_sample_length'], 1.0)
            if index < size:
                break
            index -= size
        audio_filename = os.path.join(self.unannotated_dir, file)
        offset = index * self.config['wav_sample_length']
        norm_spec = get_normalized_spectrogram(
            audio_filename, self.config, offset, self.config['wav_sample_length'])
        y = [0] * int(self.config['wav_sample_length'] / self.config['chunk_length'])
        if flipped:
            norm_spec, y = flip(norm_spec, y)
        if freq_masked:
            norm_spec = freq_mask(norm_spec, self.config['augmentation_freq_masking'])
        if time_masked:
            norm_spec = time_mask(norm_spec, self.config['augmentation_time_masking'])
        if noise_gauss:
            norm_spec = gaussian(norm_spec, self.config['augmentation_gaussian_std'])
        if permutation:
            multiply_step = int(self.config['chunk_length']  / self.config['hop_length'])
            norm_spec, y = shuffle_events(norm_spec, [i for i in y for _ in range(multiply_step)])
            y = y[::multiply_step]
        if shuffle:
            multiply_step = int(self.config['chunk_length']  / self.config['hop_length'])
            norm_spec, y = shuffle_intervals(norm_spec, [i for i in y for _ in range(multiply_step)], self.config['augmentation_shuffle_intervals'])
            y = y[::multiply_step]
        X = split_spectrogram_to_windows(
            norm_spec, self.config['wav_sample_length'], self.config['overlapping'], self.config['chunk_length'])
        return np.expand_dims(np.array(X), -1)

    def get_data(self, processes=None, unannotated=False, augmented=False):
        """Gets whole dataset.

        Returns:
            ndarray: Array of dataset.
        """
        pool = Pool(processes)
        if unannotated and not augmented:
            unannotated_length = self.getlen_unannotated()
            data = pool.map(self.getitem_unannotated, list(range(unannotated_length)))
            return np.array(data)
        if unannotated and augmented:
            unannotated_length = self.getlen_unannotated()
            weak_data = pool.starmap(self.getitem_unannotated, [[i,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                False] 
                                                                for i in range(self.getlen_unannotated())])
            strong_data = pool.starmap(self.getitem_unannotated, [[i,
                                                                False,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                True]
                                                                for i in range(self.getlen_unannotated())])
            return np.array(weak_data), np.array(strong_data)
        length = self.getlen()
        data = pool.map(self.getitem, list(range(length)))
        if self.augmentation:
            augmented_length = int(length * self.config['augmentation_size'])
            data += pool.starmap(self.getitem, [[np.random.randint(length),
                                                      is_true(self.config['augmentation_vertical_flip_probability']),
                                                      is_true(self.config['augmentation_freq_probability']),
                                                      is_true(self.config['augmentation_gaussian_probability']),
                                                      is_true(self.config['augmentation_permutation_probability']),
                                                      is_true(self.config['augmentation_shuffle_probability']),
                                                      is_true(self.config['augmentation_time_probability'])] 
                                                     for _ in range(augmented_length)])
            mixup_len = int(length * self.config['augmentation_mixup_size'])
            mixup_data = pool.starmap(self.getitem, [[np.random.randint(length)]
                                                    #   is_true(self.config['augmentation_vertical_flip_probability']),
                                                    #   is_true(self.config['augmentation_freq_probability']),
                                                    #   is_true(self.config['augmentation_gaussian_probability']),
                                                    #   is_true(self.config['augmentation_permutation_probability']),
                                                    #   is_true(self.config['augmentation_shuffle_probability']),
                                                    #   is_true(self.config['augmentation_time_probability'])] 
                                                     for _ in range(2 * mixup_len)])
            for i in range(mixup_len):
                data.append(mixup(mixup_data[i], mixup_data[2 * i], alpha=8))
        X_data, y_data = zip(*data)
        return np.array(X_data), np.array(y_data)

    def __get_categories(self, df, file_index):
        categories = []
        sample_start = file_index * self.config['wav_sample_length'] / self.sample_overlapping
        sample_end = sample_start + self.config['wav_sample_length']
        start = sample_start
        end = start + self.config['chunk_length']
        i = 0
        for _, row in df.iterrows():
            while start < row['end'] and end <= sample_end + 1e-8:
                if min(row['end'] - start, end - row['start']) > 0.5 * self.config['chunk_length']:
                    categories.append(1)
                else:
                    categories.append(0)
                i += 1
                start = sample_start + i * self.config['chunk_length'] / self.config['overlapping']
                end = start + self.config['chunk_length']
        categories += int((sample_end + 1e-8 - end) // (self.config['chunk_length'] / self.config['overlapping']) + 1) * [0]
        return categories
