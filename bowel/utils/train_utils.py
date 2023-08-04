import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score


def get_classes_amount(y):
    """Calculates classes balance.

    Args:
        y (ndarray): Array of {0,1} classes.

    Returns:
        (int, int, float): Tuple of amount of 0s, amount of 1s and ratio of 0s to all.
    """
    zeros = np.count_nonzero(y == 0)
    ones = np.count_nonzero(y == 1)
    return zeros, ones, zeros / (zeros + ones)


def get_confusion_matrix(y_true, y_pred, threshold):
    """Calculates confusion matrix.

    Args:
        y_true (ndarray): Array of ground truth classes.
        y_pred (ndarray): Array of predicted classes probabilities.
        threshold (float): Threshold in range [0,1] above which, set probabilities to 1 class.

    Returns:
        ndarray: 2D array with confusion matrix values.
    """
    y_pred = (y_pred >= threshold)
    return confusion_matrix(y_true.flatten(), (y_pred >= threshold).flatten())


def get_score(y_true, y_pred, threshold=0.5):
    """Calculates various metrics.

    Args:
        y_true (ndarray): Array of ground truth classes.
        y_pred (ndarray): Array of predicted classes probabilities.
        threshold (float): Threshold in range [0,1] above which, set probabilities to 1 class. Defaults to 0.5.

    Returns:
        dict: Dictionary with calculated metrics.
    """
    y_true = (y_true >= 0.5)
    c = get_confusion_matrix(y_true, y_pred, threshold)
    zeros, ones, ratio = get_classes_amount(y_true)
    precision = c[1, 1] / (c[1, 1] + c[0, 1])
    recall = c[1, 1] / (c[1, 1] + c[1, 0])
    return {1: ones, 0: zeros, 'ratio': ratio,
            'TN': c[0][0], 'FP': c[0][1], 'FN': c[1][0], 'TP': c[1][1],
            'accuracy': (c[0, 0] + c[1, 1]) / (c[0, 0] + c[0, 1] + c[1, 0] + c[1, 1]),
            'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall),
            'specificity': c[0, 0] / (c[0, 0] + c[0, 1]), 'auc_pr': average_precision_score(y_true.flatten(), y_pred.flatten())}

def get_scores_mean(scores):
    """Calculates average metrics from list of metrics from crossvalidation.

    Args:
        scores (list[dict]): List of metrics.

    Returns:
        dict: Dictionary with averaged metrics.
    """
    mean_score = {}
    for key in scores[0].keys():
        mean_score[key] = sum(d[key] for d in scores) / len(scores)
    return mean_score

def get_times(df, offset, duration):
    """Gets times when sounds occured from data frame with annotations.

    Args:
        df (pd.DataFrame): Data frame with columns: start, end.
        offset (float): Time in seconds from which times are taken.
        duration (float): Duration in seconds to take times.

    Returns:
        list[dict]: List of dicts with start and end values in seconds.
    """
    df = df[(df['end'] > offset) & (df['start'] < offset + duration)]
    df_times = df[['start', 'end']] - offset
    return df_times.to_dict('records')

def convert_to_sounds(frames, threshold=0.5):
        """Convert model probability predictions to sounds intervals

        Args:
            frames (list[dict]): List of dicts with start and end of times in seconds and probability of occuring a sound.
            threshold (float): Threshold value in range [0,1] above which, classify time range as sound.

        Returns:
            list[dict]: List of dicts with start and end of times in seconds
        """
        sounds = []
        is_sound = False
        start = 0.0
        for window in frames:
            if window['probability'] > threshold:
                if not is_sound:
                    is_sound = True
                    start = window['start']
            elif is_sound:
                is_sound = False
                sounds.append({'start': start, 'end': window['start']})
        if is_sound:
            sounds.append({'start': start, 'end': window['end']})
        return sounds

def get_jaccard_scores_table(files, model, threshold=0.5):
    metrics = {i * 0.1: {"TP": 0, "FN": 0, "FP": 0} for i in range(1,10)}
    for file in files:
        frames, _ = model.infer(file)
        sounds_pred = convert_to_sounds(frames, threshold)
        sounds_true = pd.read_csv(file.replace('.wav', '.csv')).to_dict('records')
        for jaccard_threshold in metrics:
            for sound_true in sounds_true:
                if any(get_jaccard_index(sound_true['start'], sound_true['end'], sound_pred['start'], sound_pred['end']) >= jaccard_threshold for sound_pred in sounds_pred):
                    metrics[jaccard_threshold]["TP"] += 1
                else:
                    metrics[jaccard_threshold]["FN"] += 1
            for sound_pred in sounds_pred:
                if not any(get_jaccard_index(sound_true['start'], sound_true['end'], sound_pred['start'], sound_pred['end']) >= jaccard_threshold for sound_true in sounds_true):
                    metrics[jaccard_threshold]["FP"] += 1
    for jaccard_threshold in metrics:
        if metrics[jaccard_threshold]["TP"] == 0:
            metrics[jaccard_threshold]["precision"] = 'NaN'
            metrics[jaccard_threshold]["recall"] = 'NaN'
            metrics[jaccard_threshold]["f1"] = 'Nan'
        else:
            metrics[jaccard_threshold]["precision"] = metrics[jaccard_threshold]["TP"] / (metrics[jaccard_threshold]["TP"] + metrics[jaccard_threshold]["FP"])
            metrics[jaccard_threshold]["recall"] = metrics[jaccard_threshold]["TP"] / (metrics[jaccard_threshold]["TP"] + metrics[jaccard_threshold]["FN"])
            metrics[jaccard_threshold]["f1"] = 2 * metrics[jaccard_threshold]["precision"] * metrics[jaccard_threshold]["recall"] / (metrics[jaccard_threshold]["precision"] + metrics[jaccard_threshold]["recall"])
    return metrics

def get_jaccard_scores(files, model, jaccard_threshold=0.5, threshold=0.5, table=False):
    metrics = {"TP": 0, "FN": 0, "FP": 0}
    for file in files:
        frames, _ = model.infer(file)
        sounds_pred = convert_to_sounds(frames, threshold)
        sounds_true = pd.read_csv(file.replace('.wav', '.csv')).to_dict('records')
        for sound_true in sounds_true:
            if any(get_jaccard_index(sound_true['start'], sound_true['end'], sound_pred['start'], sound_pred['end']) >= jaccard_threshold for sound_pred in sounds_pred):
                metrics["TP"] += 1
            else:
                metrics["FN"] += 1
        for sound_pred in sounds_pred:
            if not any(get_jaccard_index(sound_true['start'], sound_true['end'], sound_pred['start'], sound_pred['end']) >= jaccard_threshold for sound_true in sounds_true):
                metrics["FP"] += 1
    if metrics["TP"] == 0:
        metrics["precision"] = 'NaN'
        metrics["recall"] = 'NaN'
        metrics["f1"] = 'Nan'
    else:
        metrics["precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
        metrics["recall"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    return metrics


def get_jaccard_index(x1, x2, y1, y2):
    if y1 > x2 or x1 > y2:
        return 0
    return (min(x2, y2) - max(x1, y1)) / (max(x2, y2) - min(x1, y1))
