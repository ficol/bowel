import os
import argparse
import shutil

import yaml


class DataProcessor:
    """A class to preprocess raw data.
    """
    def __init__(self, raw_dir, interim_dir, processed_dir, config):
        """DataProcessor constructor.

        Args:
            raw_dir (str): Path to directory with raw data.
            interim_dir (str): Path to directory to save interim data.
            processed_dir (str): Path to directory to save processed data.
            config (dict): Dictionary with config parameters.
        """
        self.config = config
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        self.processed_dir = processed_dir
        self.train_dir = os.path.join(processed_dir, 'train')
        self.valid_dir = os.path.join(processed_dir, 'valid')
        self.test_dir = os.path.join(processed_dir, 'test')
        os.mkdir(self.train_dir)
        os.mkdir(self.valid_dir)
        os.mkdir(self.test_dir)
        os.mkdir(os.path.join(self.train_dir, 'unannotated'))

    def process_data(self):
        """Process data from raw to processed.
        """
        for input_wav in self.config['train_files']:
            input_dict = input_wav.replace('.wav', '.csv')
            shutil.copy2(os.path.join(self.raw_dir, input_wav), self.train_dir)
            shutil.copy2(os.path.join(self.raw_dir, input_dict), self.train_dir)
        for input_wav in self.config['valid_files']:
            input_dict = input_wav.replace('.wav', '.csv')
            shutil.copy2(os.path.join(self.raw_dir, input_wav), self.valid_dir)
            shutil.copy2(os.path.join(self.raw_dir, input_dict), self.valid_dir)
        for input_wav in self.config['test_files']:
            input_dict = input_wav.replace('.wav', '.csv')
            shutil.copy2(os.path.join(self.raw_dir, input_wav), self.test_dir)
            shutil.copy2(os.path.join(self.raw_dir, input_dict), self.test_dir)
        if 'unannotated_files' in self.config:
            for unannotated_wav in self.config['unannotated_files']:
                shutil.copy2(os.path.join(os.path.join(self.raw_dir, 'unannotated'), unannotated_wav), os.path.join(self.train_dir, 'unannotated'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Path to raw directory')
    parser.add_argument('--interim_dir', type=str, default='data/interim',
                        help='Path to interim directory')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Path to processed directory')
    parser.add_argument('--config', type=str, default='bowel/configs/crnnconfig.yml',
                        help='Path to yaml file with data and model parameters')
    args = parser.parse_args()

    data_processor = DataProcessor(
        args.raw_dir, args.interim_dir, args.processed_dir, yaml.safe_load(open(args.config)))
    data_processor.process_data()
