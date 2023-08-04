import argparse

import yaml
import uuid
import pandas as pd

from bowel.models.crnn import CRNN
from bowel.models.conv_merge import ConvMerge
from bowel.models.rnn import RNN
from bowel.models.logistic_regression import LogisticRegression
from bowel.models.svm import SVM
from bowel.models.random_forest import RandomForest
from bowel.models.gradient_boosting import GradientBoosting
from bowel.utils.train_utils import convert_to_sounds

class Inference:
    """A class to inference and save predictions.
    """

    def __init__(self, model_file, config):
        """Inference constructor.

        Args:
            model_file (str): Path to trained model.
            config (str): Path to config with model parameters.
        """
        self.config = config
        if config['model_type'] == 'convrnn':
            self.model = CRNN(config, model_file=model_file)
        if config['model_type'] == 'convmerge':
            self.model = ConvMerge(config, model_file=model_file)
        if config['model_type'] == 'convrnnnarrow':
            self.model = RNN(config, model_file=model_file)
        if config['model_type'] == 'logistic_regression':
            self.model = LogisticRegression(config, model_file=model_file)
        if config['model_type'] == 'random_forest':
            self.model = RandomForest(config, model_file=model_file)
        if config['model_type'] == 'svm':
            self.model = SVM(config, model_file=model_file)
        if config['model_type'] == 'gradient_boosting':
            self.model = GradientBoosting(config, model_file=model_file)

    def infer(self, wav_file):
        """Inference on audio file

        Args:
            wav_file (str): Path to audio file.

        Returns:
            list[dict]: List of dicts with start and end of times in seconds and probability of occuring a sound.
            float: duration of wav file.
        """
        return self.model.infer(wav_file)

    def save_predictions(self, csv_file, sounds, duration):
        """Save predictions to csv file.

        Args:
            csv_file (str): Path to csv file to save predictions.
        """
        df = pd.DataFrame(sounds)
        with open(csv_file, 'w') as f:
            f.write(f'# inference_hash={uuid.uuid1().hex}\n')
            f.write(f'# model_type={self.config["model_type"]}\n')
            f.write(f'# model_version={self.config["model_version"]}\n')
            f.write(f'# duration={round(duration, 3)}\n')
        df.to_csv(csv_file, index=False, float_format='%.3f', mode='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Path to saved model')
    parser.add_argument('--wav_file', type=str,
                        help='Path to wav file')
    parser.add_argument('--csv_output', type=str,
                        help='Path to csv file to save')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Threshold value in range [0,1] above which, classify time range as sound.')
    args = parser.parse_args()

    inference_model = Inference(args.model, yaml.safe_load(open(args.model.replace('.h5', '.yml'))))
    frames, duration = inference_model.infer(args.wav_file)
    sounds = convert_to_sounds(frames, args.threshold)
    inference_model.save_predictions(args.csv_output, sounds, duration)
