import os
import argparse

import yaml
import numpy as np
import wandb

from bowel.models.crnn import CRNN
from bowel.models.rnn import RNN
from bowel.models.conv_merge import ConvMerge
from bowel.models.curriculum_labeling import CurriculumLabel
from bowel.models.fixmatch import FixMatch
from bowel.models.logistic_regression import LogisticRegression
from bowel.models.random_forest import RandomForest
from bowel.models.svm import SVM
from bowel.models.gradient_boosting import GradientBoosting
from bowel.data.sequence_load import SequenceLoader
from bowel.data.neighbor_load import NeighborLoader
from bowel.utils.train_utils import get_score, get_jaccard_scores, get_jaccard_scores_table
from bowel.config import *


class Trainer:
    """A class to train and test models.
    """

    def __init__(self, data_dir, model_file, config):
        """Trainer constructor.

        Args:
            data_dir (str): Path to directory with processed data.
            model_file (str): Path to model file.
            config (dict): Dictionary with config parameters.
        """
        self.config = config
        self.train_dir = os.path.join(data_dir, 'train')
        self.valid_dir = os.path.join(data_dir, 'valid')
        self.test_dir = os.path.join(data_dir, 'test')
        self.model_file = model_file
        self.semisupervised = False
        if 'semisupervised' in config and config['semisupervised'] == 'curriculum':
            self.semisupervised = True
            self.Model = CurriculumLabel
            self.Load = SequenceLoader
        elif 'semisupervised' in config and config['semisupervised'] == 'fixmatch':
            self.semisupervised = True
            self.Model = FixMatch
            self.Load = SequenceLoader
        elif config['model_type'] == 'convrnn':
            self.Model = CRNN
            self.Load = SequenceLoader
        elif config['model_type'] == 'convmerge':
            self.Model = ConvMerge
            self.Load = SequenceLoader
        elif config['model_type'] == 'convrnnnarrow':
            self.Model = RNN
            self.Load = SequenceLoader
        elif config['model_type'] == 'logistic_regression':
            self.Model = LogisticRegression
            self.Load = NeighborLoader
        elif config['model_type'] == 'random_forest':
            self.Model = RandomForest
            self.Load = NeighborLoader
        elif config['model_type'] == 'svm':
            self.Model = SVM
            self.Load = NeighborLoader
        elif config['model_type'] == 'gradient_boosting':
            self.Model = GradientBoosting
            self.Load = NeighborLoader

    def train(self):
        """Train model, print metrics on trained model and save model to file.
        """
        train_loader = self.Load(self.train_dir, self.config, augmentation=True)
        valid_loader = self.Load(self.valid_dir, self.config)
        X_train, y_train = train_loader.get_data()
        X_valid, y_valid = valid_loader.get_data()  
        if self.semisupervised:
            model = self.Model(self.config, X_train[0].shape)
            print(model.summary())
            if self.config['semisupervised'] == 'curriculum':
                X_unannotated = train_loader.get_data(unannotated=True)
                model.train(X_train, y_train, X_unannotated, X_valid, y_valid)
            elif self.config['semisupervised'] == 'fixmatch':
                weak_X_unannotated, strong_X_unannotated = train_loader.get_data(unannotated=True, augmented=True)
                model.train(X_train, y_train, weak_X_unannotated, strong_X_unannotated, X_valid, y_valid)
        else:
            model = self.Model(self.config, X_train[0].shape)
            print(model.summary())
            model.train(X_train, y_train, X_valid, y_valid)
        model.save(self.model_file)
        y_train_pred = model.predict(X_train)
        train_metrics = get_score(y_train, y_train_pred)
        train_iou_metrics = get_jaccard_scores([os.path.join(self.train_dir, file) for file in os.listdir(self.train_dir) if file.endswith('.wav')], model)
        results_file = os.path.splitext(self.model_file)[0] + '.txt'
        with open(results_file, 'w') as f:
            f.write('train data:\n')
            for metric in train_metrics:
                f.write(f'    {metric} = {train_metrics[metric]}\n')
            for iou_metric in train_iou_metrics:
                f.write(f'    IOU {iou_metric} = {train_iou_metrics[iou_metric]}\n')
            y_valid_pred = model.predict(X_valid)
            valid_metrics = get_score(y_valid, y_valid_pred)
            valid_iou_metrics = get_jaccard_scores([os.path.join(self.valid_dir, file) for file in os.listdir(self.valid_dir) if file.endswith('.wav')], model)
            f.write('valid data:\n')
            for metric in valid_metrics:
                f.write(f'    {metric} = {valid_metrics[metric]}\n')
            for iou_metric in valid_iou_metrics:
                f.write(f'    IOU {iou_metric} = {valid_iou_metrics[iou_metric]}\n')
        with open(results_file, 'r') as f:
            print(f.read())

    def test(self, reports_dir):
        """Print metrics on loaded trained model.
        """
        train_loader = self.Load(self.train_dir, self.config)
        valid_loader = self.Load(self.valid_dir, self.config)
        test_loader = self.Load(self.test_dir, self.config)
        X_train, y_train = train_loader.get_data()
        X_valid, y_valid = valid_loader.get_data()
        X_test, y_test = test_loader.get_data()
        model = self.Model(self.config, X_train[0].shape, self.model_file)
        threshold=0.5
        y_train_pred = model.predict(X_train)
        train_metrics = get_score(y_train, y_train_pred, threshold)
        train_iou_metrics = get_jaccard_scores_table([os.path.join(self.train_dir, file) for file in os.listdir(self.train_dir) if file.endswith('.wav')], model, threshold)
        y_valid_pred = model.predict(X_valid)
        valid_metrics = get_score(y_valid, y_valid_pred)
        valid_iou_metrics = get_jaccard_scores_table([os.path.join(self.valid_dir, file) for file in os.listdir(self.valid_dir) if file.endswith('.wav')], model, threshold)
        y_test_pred = model.predict(X_test)
        test_metrics = get_score(y_test, y_test_pred)
        test_iou_metrics = get_jaccard_scores_table([os.path.join(self.test_dir, file) for file in os.listdir(self.test_dir) if file.endswith('.wav')], model, threshold)
        reports_file = os.path.join(reports_dir, os.path.splitext(os.path.basename(self.model_file))[0] + '.txt')
        with open(reports_file, 'w') as f:
            f.write(f'threshold={threshold}\n')
            f.write('train data:\n')
            for metric in train_metrics:
                f.write(f'    {metric} = {train_metrics[metric]}\n')
            f.write('valid data:\n')
            for metric in valid_metrics:
                f.write(f'    {metric} = {valid_metrics[metric]}\n')
            f.write('test data:\n')
            for metric in test_metrics:
                f.write(f'    {metric} = {test_metrics[metric]}\n')
            for jaccard_threshold in train_iou_metrics:
                f.write(f'IOU threshold = {jaccard_threshold}\n')
                f.write('train data:\n')
                for iou_metric in train_iou_metrics[jaccard_threshold]:
                    f.write(f'    IOU {iou_metric} = {train_iou_metrics[jaccard_threshold][iou_metric]}\n')
                f.write('valid data:\n')
                for iou_metric in valid_iou_metrics[jaccard_threshold]:
                    f.write(f'    IOU {iou_metric} = {valid_iou_metrics[jaccard_threshold][iou_metric]}\n')
                f.write('test data:\n')
                for iou_metric in test_iou_metrics[jaccard_threshold]:
                    f.write(f'    IOU {iou_metric} = {test_iou_metrics[jaccard_threshold][iou_metric]}\n')
        with open(reports_file, 'r') as f:
            print(f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='"train": training model, "test": testing model on dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='path to data directory')
    parser.add_argument('--reports_dir', type=str, default='reports',
                        help='path to reports directory')
    parser.add_argument('--model', type=str, default='models/model.h5',
                        help='if "train" and "crossval" mode: path to save model, if "test": path to load model')
    parser.add_argument('--config', type=str, default='bowel/configs/crnnconfig.yml',
                        help='yaml file with data and model parameters. only for train mode.')
    parser.add_argument('--dryrun', action='store_true',
                        help='not sync run with wandb')
    parser.add_argument('--experiment', type=str, default=None,
                        help='name of wandb experiment')
    args = parser.parse_args()
    np.random.seed(10)

    if args.mode == 'train':
        config = yaml.safe_load(open(args.config))
        if args.dryrun:
            os.environ['WANDB_MODE'] = 'dryrun'
        if args.experiment is not None:
            wandb.init(project=WANDB_PROJECT_NAME, config=config, name=args.experiment, tags=["important"])
        else:
            wandb.init(project=WANDB_PROJECT_NAME, config=config)
        trainer = Trainer(args.data_dir, args.model, config)
        trainer.train()
    elif args.mode == 'test':
        config = yaml.safe_load(open(args.model.replace('.h5', '.yml')))
        trainer = Trainer(args.data_dir, args.model, config)
        trainer.test(args.reports_dir)
