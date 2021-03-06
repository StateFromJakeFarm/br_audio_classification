import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import logging
import argparse

from UrbanSoundDataManager import UrbanSoundDataManager
from SoundSortDataManager import SoundSortDataManager
from sys import argv, stderr
from os.path import join, isfile
from collections import OrderedDict
from random import shuffle

class Classifier:
    '''
    Train LSTM models to identify each class of the
    UrbanSound8K dataset
    '''
    class Model(nn.Module):
        '''
        Model used to identify each class
        '''
        def init_state_tensors(self):
            self.h = torch.zeros(self.num_recurrent, self.batch_size,
                self.hidden_size).to(self.device)

        def __init__(self, input_size, hidden_size, batch_size, num_recurrent,
            dropout, chunks, chunk_len, label, device):
            super(Classifier.Model, self).__init__()

            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.num_recurrent = num_recurrent
            self.chunks = chunks
            self.chunk_len = chunk_len
            self.label = label
            self.device = device

            # Define model
            self.preprocessor = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
            )
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_recurrent, dropout=dropout)
            self.postprocessor = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

            # Init hidden and cell states
            self.init_state_tensors()

        def forward(self, x):
            x = self.preprocessor(x)
            x, self.h = self.gru(x, self.h)
            return self.postprocessor(x[-1])

    def __init__(self, dataset_path, hidden_size, batch_size, num_recurrent,
        lr, dropout, sr, file_duration, device_id, train_class_pct,
        save, gathered, max_files=None):
        logging.info('Initializing classifier')
        self.batch_size = batch_size

        # Whether or not we'll be saving snapshots of models during training
        self.save_dir = save
        if self.save_dir is not None:
            # Log script output to file
            full_save_path = join('saved_models', self.save_dir)
            os.makedirs(full_save_path)

            handler = logging.FileHandler(join(full_save_path, 'output.log'))
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

            # Save model parameters so they can be loaded during testing
            model_params = {
                'hidden_size': hidden_size,
                'batch_size': batch_size,
                'num_recurrent': num_recurrent,
                'sr': sr,
                'file_duration': file_duration,
                'gathered': gathered,
                'dropout': dropout,
            }
            pickle.dump(model_params, open(join(full_save_path, 'model_params.p'), 'wb'))

        # Init data manager
        logging.info('Initializing data manager')
        if gathered is not None:
            # Use dataset gathered by scraper
            if os.environ.get('SOUNDSCRAPE_AUTH_JSON') is None:
                raise EnvironmentError('SOUNDSCRAPE_AUTH_JSON must be an environment variable containing path to service account JSON credentials file')
            self.dm = SoundSortDataManager(dataset_path, os.environ['SOUNDSCRAPE_AUTH_JSON'], 'soundscrape-bucket', gathered.split(','), batch_size=self.batch_size,
                train_class_pct=train_class_pct, file_duration=file_duration, sr=sr, max_files=max_files)
        else:
            # Use UrbanSound8K dataset
            self.dm = UrbanSoundDataManager(join(dataset_path, 'audio'), batch_size=self.batch_size, train_class_pct=train_class_pct, file_duration=file_duration, sr=sr)

        # Loss function used during training
        self.loss_function = nn.MSELoss()

        # Determine device for training
        self.use_cuda = (torch.cuda.device_count() >= 1)
        if self.use_cuda and device_id is not None:
            # Use GPU if available
            logging.info('Using CUDA GPU')
            device_id = int(device_id)
            self.device = torch.device('cuda:{}'.format(device_id))
        else:
            # Otherwise use CPU
            logging.info('Using CPU')
            self.use_cuda = False
            self.device = torch.device('cpu')

        logging.info('Constructing models')
        self.models = []
        self.optimizers = []
        for label in range(len(self.dm.classes)):
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, num_recurrent, dropout,
            self.dm.chunks, self.dm.chunk_len, label, self.device)

            model.float()

            if self.use_cuda:
                # Prep model to train on GPU
                label = model.label
                model = nn.DataParallel(model, device_ids=[device_id])

                # Hack to get DataParallel object to maintain Model object's label
                setattr(model, 'label', label)

            self.models.append(model)

            # Optimizers specific to each model
            self.optimizers.append(optim.Adam(model.parameters(), lr=lr))

            if self.save_dir is not None:
                # Create model's save directory
                os.makedirs(join('saved_models', self.save_dir, str(label)))

    def test(self, model, save_path=None):
        '''
        Determine a model's accuracy against testing set
        '''
        missed_this_class, missed_other_classes, total_this_class = 0, 0, 0
        with torch.no_grad():
            model.eval()
            for batch, labels in self.dm.testing_batches:
                # Get testing batch
                batch.to(self.device)
                labels_tensor = torch.Tensor([float(label == model.label) for label in labels]).to(self.device)

                # Wipe state clean for next file (gross way to do it)
                try:
                    model.module.init_state_tensors()
                except AttributeError:
                    model.init_state_tensors()

                # Run model
                output = model(batch)

                # Calculate accuracy
                output_rounded = torch.round(output.t())
                diff = (labels_tensor - output_rounded)

                for i, v in enumerate(diff[0]):
                    v = v.item()
                    total_this_class += int(labels_tensor[i])
                    if abs(v):
                        # Missed this one
                        if labels_tensor[i]:
                            missed_this_class += 1
                        else:
                            missed_other_classes += 1

        if save_path is not None:
            # Save model
            torch.save(model.state_dict(), save_path)

        return total_this_class - missed_this_class, total_this_class, missed_other_classes, len(self.dm.testing_batches)*self.batch_size - total_this_class

    def train(self, epochs):
        logging.info('Begin training ({} epochs)'.format(epochs))

        # Train each model serially to minimize memory footprint
        for i, model in enumerate(self.models):
            logging.info('Training "{}" model ({}/{})'.format(self.dm.classes[model.label], i+1, len(self.models)))

            # Move model to training device
            model.to(self.device)

            # Retrieve optimizer
            optimizer = self.optimizers[i]

            # Load training set for this class into memory
            self.dm.load_training_batches(i)

            for e in range(epochs):
                shuffle(self.dm.training_batches)
                for batch, labels in self.dm.training_batches:
                    # Retrieve batch
                    batch.to(self.device)

                    # Wipe state clean for next file (gross way to do it)
                    try:
                        model.module.init_state_tensors()
                    except AttributeError:
                        model.init_state_tensors()

                    # Run network
                    output = model(batch)

                    # Determine correct output for this batch
                    correct_output = torch.reshape(torch.Tensor([float(label == model.label) for label in labels]), (self.batch_size, 1)).to(self.device)

                    # Calculate loss
                    loss = self.loss_function(output, correct_output)

                    # Backpropagate
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                if (e+1) % (epochs//10) == 0:
                    # Run against test set
                    if self.save_dir is not None:
                        # Save a snapshot
                        save_path = join('saved_models', self.save_dir, str(model.label), '{}_epochs.pth'.format(e+1))
                    else:
                        save_path = None

                    pos_this_class, total_this_class, pos_other_classes, total_other_classes = self.test(model, save_path=save_path)
                    model.train()
                    logging.info('  ({}/{}) correctly identified for this class: {}/{}  false positives: {}/{}'.format(
                        e+1, epochs, pos_this_class, total_this_class, pos_other_classes, total_other_classes))

        logging.info('Finish training')

    def get_latest_save_file(self, saved_classifier_path):
        '''
        Return path to the latest (most training epochs) save file for this
        model
        '''
        files = os.listdir(saved_classifier_path)

        return join(saved_classifier_path, max(files, key=lambda f: int(f.split('_')[0])))

    def full_test(self, saved_classifiers_path=None):
        '''
        Test entire system by passing entire test set through every classifier,
        then choosing maximum output value from each classifier as each file's
        label
        '''
        logging.info('Testing accuracy of entire system')

        # Determine number of test files and batches to be used
        total_test_files = len(self.dm.test_files)
        num_test_files = total_test_files - (total_test_files % self.batch_size)

        # Record all output
        output = torch.zeros([len(self.dm.classes), num_test_files], dtype=torch.float32)
        labels = torch.zeros([num_test_files], dtype=torch.int16)

        # Load and test each model
        for i, c in enumerate(self.dm.classes):
            logging.info('Running "{}" model'.format(c))

            model = self.models[i]
            if saved_classifiers_path is not None:
                # Get latest save file for this model
                save_file = self.get_latest_save_file(join(saved_classifiers_path, str(i)))
                logging.debug('Loading {}'.format(save_file))

                # Load model parameters
                state_dict = torch.load(save_file, map_location=self.device)
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError:
                    # Model (probably) trained with GPU and we're now trying to load it with a CPU,
                    # which requires restructuring state dictionary
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[ k[7:] ] = v

                    model.load_state_dict(new_state_dict)

            # Collect all output for testing set
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                for j, batch_and_labels_tuple in enumerate(self.dm.testing_batches):
                    batch, batch_labels = batch_and_labels_tuple
                    batch.to(self.device)

                    # Wipe state clean for next file (gross way to do it)
                    try:
                        model.module.init_state_tensors()
                    except AttributeError:
                        model.init_state_tensors()

                    # Record output and labels for this batch
                    output[i][j*self.batch_size:j*self.batch_size+self.batch_size] = model(batch).t()
                    if i == 0:
                        # No need to overwrite labels with the same data on every pass
                        labels[j*self.batch_size:j*self.batch_size+self.batch_size] = torch.Tensor(batch_labels)

        # Calculate accuracy
        output = output.t()
        correct = [0 for c in self.dm.classes]
        correct_negatives = 0
        total = [0 for c in self.dm.classes]
        for i in range(num_test_files):
            label = labels[i].item()

            ensemble_label = output[i].argmax().item()
            if label < len(self.dm.classes):
                total[label] += 1
                correct[label] += (ensemble_label == label)
            else:
                correct_negatives += (ensemble_label < 0.5)

        logging.info('Accuracy by class:')
        for i, class_name in enumerate(self.dm.classes):
            logging.info('  {}: {}/{} correctly labeled ({:.2}%)'.format(
                class_name, correct[i], total[i], float(correct[i])/total[i]*100))

        total_correct = sum(correct) + correct_negatives

        logging.info('Overall performance: {}/{} ({:.2f}% accuracy)'.format(total_correct, num_test_files, float(total_correct)/num_test_files*100))

if __name__ == '__main__':
    # Create argument parser for easier CLI
    parser = argparse.ArgumentParser(description='Ensemble of binary relevance models for audio classification',
        prog='classifier.py')
    parser.add_argument('-p', '--path', type=str, required=True,
        help='path to UrbanSound[8K] dataset OR soundScrape download cache')
    parser.add_argument('--hidden', type=int, default=128,
        help='dimension of hidden internal layers of network (default is 128)')
    parser.add_argument('-b', '--batch', type=int, default=100,
        help='batch size for training and testing (default is 100)')
    parser.add_argument('--lr', type=float, default=0.005,
        help='learning rate (default is 0.005)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
        help='number of training epochs (default is 100)')
    parser.add_argument('-t', '--target', type=float, default=0.5,
        help='fraction of training set to be of target class; accomplished by resampling target class to correct imbalance (default is 0.5)')
    parser.add_argument('-d', '--duration', type=float, default=2,
        help='number of seconds used from each file in dataset; those shorter are zero-padded (default is 2)')
    parser.add_argument('-r', '--recurrent', type=int, default=3,
        help='number of recurrent layers in each model (default is 3)')
    parser.add_argument('--sr', type=int, default=16000,
        help='sample rate at which files are loaded (default is 16000)')
    parser.add_argument('--dropout', type=float, default=0.0,
        help='drouput rate for recurrent part of network during training (default is 0)')
    parser.add_argument('-s', '--save', type=str, default=None,
        help='name of save directory (cannot already exist in "saved_models/")')
    parser.add_argument('-c', '--card', type=int, default=None,
        help='index of CUDA-capable GPU to be used for training and testing')
    parser.add_argument('-g', '--gathered', type=str, default=None,
        help='comma-separated classes to identify within dataset gathered by soundScrape')
    parser.add_argument('-m', '--max', type=int, default=None,
        help='maximum number of files to use in training and testing sets')
    parser.add_argument('-l', '--load', type=str, default=None,
        help='path to classifier save directory to be loaded and tested')
    args = parser.parse_args()

    # Set log level to info
    logging.getLogger().setLevel(logging.INFO)

    if args.load is not None:
        logging.info('Loading model from save directory')

        model_params = {}
        model_params_path = join(args.load, 'model_params.p')
        if isfile(model_params_path):
            model_params = pickle.load(open(model_params_path, 'rb'))

            classifier = Classifier(args.path, model_params['hidden_size'], model_params['batch_size'],
                model_params['num_recurrent'], args.lr, model_params['dropout'], model_params['sr'],
                model_params['file_duration'], args.card, args.target, args.save, model_params['gathered'],
                args.max)

            # Test entire system by running test set through every classifier
            classifier.full_test(saved_classifiers_path=args.load)
        else:
            logging.error('Could not find "model_params.p" in save directory')
    else:
        classifier = Classifier(args.path, args.hidden, args.batch, args.recurrent,
            args.lr, args.dropout, args.sr, args.duration, args.card, args.target,
            args.save, args.gathered, args.max)

        # Train all classifiers to identify their respective sounds
        classifier.train(args.epochs)

        # Free up memory
        del classifier.dm.training_batches

        # Test entire ensemble
        classifier.full_test()
