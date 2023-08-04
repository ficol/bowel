import numpy as np
import random
import librosa
import soundfile


def load_samples(filename, sr=None, offset=None, duration=None, padding=False):
    """Loads audio samples from file.

    Args:
        filename (str): Path to audio file.
        sr (int, optional): Sample rate to resample audio. If None sample rate isn't changed. Defaults to None.
        offset (float, optional): Time in seconds from which Loading is started. If None offset equals 0. Defaults to None.
        duration (float, optional): Duration in seconds to load from audio file. If None duration equals audio file length. Defaults to None.

    Returns:
        (ndarray, int): Tuple of 1D array of samples and value of sample rate.
    """
    samples, sample_rate = librosa.load(
        filename, sr=sr, mono=True, offset=offset, duration=duration)
    if padding and duration is not None:
        samples = librosa.util.fix_length(samples, size=int(duration * sample_rate))
    return samples, sample_rate


def spectrogram(samples, sample_rate, fft, hop_length, window, max_freq=None):
    """Converts audio samples to spectrogram.

    Args:
        samples (ndarray): 1D array of samples.
        sample_rate (float): Sample rate.
        fft (int): FFT frame width in seconds.
        hop_length (int): Hop length in seconds.
        window (str): Window function to apply on frame.
        max_freq (int, optional): Maximal frequency to cut spectrogram values. If None do not cut. Defaults to None.

    Returns:
        ndarray: 2D array of spectrogram values.
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(
        samples, n_fft=int(fft * sample_rate), hop_length=int(hop_length * sample_rate), window=window)))
    if max_freq is not None:
        bins_amount = int(len(D) * max_freq / (sample_rate / 2))
        D = D[:bins_amount, :]
    return D


def normalized_spectrogram(D, audio_mean, audio_std):
    """Normalize spectrogram.

    Args:
        D (ndarray): 2D array of spectrogram values.
        audio_mean (float): Mean of spectrogram values used to normalize.
        audio_std (float): Standard deviation of spectrogram values used to normalize.

    Returns:
        ndarray: 2D array of normalized spectrogram values.
    """
    D -= audio_mean
    D /= audio_std
    return D


def delta_spectrogram(D):
    """Get delta spectrogram.

    Args:
        D (ndarray): 2D array of spectrogram values.

    Returns:
        ndarray: 2D array of delta spectrogram values.
    """
    return librosa.feature.delta(D)


def get_normalized_spectrogram(filename, config, offset=None, duration=None, padding=False):
    """Load audio file and convert to normalized spectrogram.

    Args:
        filename (str): Path to audio file.
        config (dict): Config parameters to calculate spectrogram.
        offset (float, optional): Time in seconds from which loading is started. If None offset equals 0. Defaults to None.
        duration (float, optional): Duration in seconds to load from audio file. If None duration equals audio file length. Defaults to None.

    Returns:
        ndarray: 2D array of normalized spectrogram values.
    """
    samples, sample_rate = load_samples(
        filename, offset=offset, duration=duration, padding=padding)
    spec = spectrogram(samples, sample_rate,
                       config['fft'], config['hop_length'], config['window_type'], config['max_freq'])
    return normalized_spectrogram(spec, config['audio_mean'], config['audio_std'])


def split_spectrogram_to_windows(spec, length, overlapping, chunk_length):
    """Split spectrogram into frames of given width.

    Args:
        spec (ndarray): 2D array of spectrogram values.
        length (float): Length of audio that spectrogram represents in seconds.
        overlapping (float): Overlapping of divided frames. 1 - no overlapping, 2 - 50% overlapping, 4 - 75% overlapping. 
        chunk_length (float): 

    Returns:
        list[ndarray]: List of 2D arrays of framed spectrogram values.
    """
    size = get_size(length, chunk_length, overlapping)
    step = int(chunk_length * spec.shape[1] / length)
    split_spectrogram = [spec[:, min(spec.shape[1], int(i * spec.shape[1] / size) + step) - step:min(
        spec.shape[1], int(i * spec.shape[1] / size) + step)] for i in range(size)]
    return split_spectrogram


def save_wav(filename, samples, sample_rate):
    """Save audio to file from samples

    Args:
        filename (str): Path to audio file to save.
        samples (ndarray): 1D array of samples.
        sample_rate (int): Sample rate.
    """
    soundfile.write(filename, samples, sample_rate)


def get_wav_length(filename):
    """Gets length of audio file.

    Args:
        filename (str): Path to audio file.

    Returns:
        float: Length of audio file in seconds.
    """
    return librosa.get_duration(filename=filename)

def get_size(length, chunk_length, overlapping):
    return int(length * overlapping / chunk_length - overlapping + 1)

def gaussian(X, std):
    return X + np.random.normal(0, std, X.shape)

def flip(X, y):
    return np.flip(X, axis=1), y[::-1]

def freq_mask(X, size):
    freqs_amount = int(X.shape[0] * size)
    freq = random.randrange(0, X.shape[0] - freqs_amount)
    X[freq:freq + freqs_amount] = 0.0
    return X

def time_mask(X, size):
    length = int(X.shape[1] * size)
    time = random.randrange(0, X.shape[1] - length)
    X[:, time:time + length] = 0.0
    return X

def mixup(sample1, sample2, alpha=0.2):
    beta = np.random.beta(alpha, alpha)
    return beta * sample1[0] + (1 - beta) * sample2[0], beta * sample1[1] + (1 - beta) * sample2[1]

def shuffle_events(X, y):
    sounds = {}
    noises = {}
    sounds_indexes = []
    if y[0] == 0:
        sound_start = -1
        noise_start = 0
    else:
        sound_start = 0
        noise_start = -1
    range_index = 0
    for i, j in enumerate(y):
        if j == 1 and noise_start != -1:
            noises[(noise_start, i)] = range_index
            range_index += 1
            sound_start = i
            noise_start = -1
        if j == 0 and sound_start != -1:
            sounds[(sound_start, i)] = range_index
            sounds_indexes.append(range_index)
            range_index += 1
            sound_start = -1
            noise_start = i
    if j == 0 and noise_start != -1:
        noises[(noise_start, i + 1)] = range_index
    if j == 1 and sound_start != -1:
        sounds[(sound_start, i + 1)] = range_index
        sounds_indexes.append(range_index)
    random.shuffle(sounds_indexes)
    for i, sound in enumerate(sounds):
        sounds[sound] = sounds_indexes[i]
    shuffled_sounds_noises = [list(range(k[0],k[1])) for k, _ in sorted((sounds | noises).items(), key=lambda x:x[1])]
    shuffled_columns = [item for sublist in shuffled_sounds_noises for item in sublist]
    return X[:, shuffled_columns], [y[i] for i in shuffled_columns]

def shuffle_intervals(X, y, intervals):
    columns = []
    for i in range(intervals):
        columns.append(list(range(int(i/intervals * len(y)), (int((i + 1)/intervals * len(y))))))
    random.shuffle(columns)
    shuffled_columns = [item for sublist in columns for item in sublist]
    return X[:, shuffled_columns], [y[i] for i in shuffled_columns]

def is_true(probability):
    return random.random() < probability
