import argparse

import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from bowel.utils.audio_utils import load_samples, spectrogram, gaussian, flip, freq_mask, time_mask, mixup, shuffle_intervals
from bowel.utils.train_utils import get_times


def plot_time_domain(samples, sample_rate, duration, ground_truth_times, predicted_times):
    """Plot audio samples.

    Args:
        samples (ndarray): 1D array of samples.
        sample_rate (int): Sample rate of audio samples.
    """
    plt.figure(figsize=(12, 3))
    for time in ground_truth_times:
        plt.axvspan(max(0, time['start']), min(duration, time['end']), facecolor='green', alpha=0.3)
    for time in predicted_times:
        plt.axvspan(max(0, time['start']), min(duration, time['end']), facecolor='red', alpha=0.3)
    librosa.display.waveshow(samples, sr=sample_rate)
    plt.ylabel('Amplitude')


def plot_spectrogram(spec, duration, ground_truth_times, predicted_times, min_freq, max_freq):
    """Plot spectrogram with marked ground truth and predicted values.

    Args:
        spec (ndarray): 2D array of spectrogram values
        duration (float): Duration of audio from which spectrogram was created.
        ground_truth_times ([list[dict]): List of ground truth times when sounds occured.
        predicted_times (list[dict]): List of predicted times when sounds occured.
        threshold (float, optional): Threshold value in range [0,1] above which, classify time range as sound. Defaults to 0.5.
    """
    plt.figure(figsize=(15, 5))
    plt.imshow(spec[::-1], aspect='auto', interpolation='none', extent=[0, duration, min_freq, max_freq])
    for time in ground_truth_times:
        plt.axvspan(max(0, time['start']), min(duration, time['end']), facecolor='red', alpha=0.3)
    for time in predicted_times:
        plt.axvspan(max(0, time['start']), min(duration, time['end']), ec='red', facecolor='none', alpha=1.0)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str,
                        help='Path to wav file')
    parser.add_argument('-o', '--offset', type=float, default=0.0,
                        help='Offset in seconds')
    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='Duration in seconds')
    parser.add_argument('--config', type=str, default='bowel/configs/crnnconfig.yml',
                        help='Path to config with spectrogram parameters')
    parser.add_argument('--truth_csv', type=str, default=None,
                        help='Path to ground truth csv')
    parser.add_argument('--predict_csv', type=str, default=None,
                        help='Path to csv with predictions')
    parser.add_argument('--time_domain', action='store_true',
                        help='Plot time domain')
    parser.add_argument('--augment', type=str, default=None,
                        help='Optional augmentation')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    samples, sample_rate = load_samples(args.filename, offset=args.offset, duration=args.duration)
    duration = len(samples) / sample_rate
    spec = spectrogram(samples, sample_rate, config['fft'], config['hop_length'], config['window_type'], config['max_freq'])
    if args.augment == 'gaussian':
        spec = gaussian(spec, 2)
    if args.augment == 'flip':
        spec, _ = flip(spec, [])
    if args.augment == 'freq_mask':
        spec = freq_mask(spec, 0.1)
    if args.augment == 'time_mask':
        spec = time_mask(spec, 0.05)
    if args.augment == 'mixup':
        spec2, _ = flip(spec, [])
        spec, _ = mixup((spec, 0), (spec2, 0))
    if args.augment == 'shuffle':
        spec, _ = shuffle_intervals(spec, spec.shape[1] * [0], 5)
    predicted_times = []
    ground_truth_times = []
    if args.truth_csv is not None:
        df = pd.read_csv(args.truth_csv, comment='#')
        ground_truth_times = get_times(df, args.offset, duration)
    if args.predict_csv is not None:
        df = pd.read_csv(args.predict_csv, comment='#')
        predicted_times = get_times(df, args.offset, duration)
    plot_spectrogram(spec, duration, ground_truth_times, predicted_times, 0, config['max_freq'])
    plt.show()
    if args.time_domain:
        plot_time_domain(samples, sample_rate, duration, ground_truth_times, predicted_times)
        plt.show()
