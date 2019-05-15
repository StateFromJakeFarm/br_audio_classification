import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse

from UrbanSoundDataManager import UrbanSoundDataManager
from SoundSortDataManager import SoundSortDataManager
from sys import argv, stderr
from os.path import join
from collections import OrderedDict

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
            self.h = torch.zeros(self.num_recurrent, self.batch_size, self.hidden_size).to(self.device)

        def __init__(self, input_size, hidden_size, batch_size, num_recurrent, chunks, chunk_len, label, device):
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
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_recurrent, batch_first=True)
            self.postprocessor = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

            # Init hidden and cell states
            self.init_state_tensors()

        def forward(self, x):
            # Run network
            x = self.preprocessor(x)
            x, self.h = self.gru(x, self.h)
            return self.postprocessor(self.h)[0]

    def __init__(self, dataset_path, hidden_size, batch_size, num_recurrent, lr, sr, file_duration, device_id, train_class_pct, min_accuracy, save, gathered):
        logging.info('Initializing classifier')
        self.batch_size = batch_size
        self.min_accuracy = min_accuracy

        # Whether or not we'll be saving snapshots of models during training
        self.save = save
        if save:
            # Create directory for saving output
            self.save_dir = '{}h_{}r_{}sr_{}'.format(hidden_size, num_recurrent, sr, 'soundScrape' if gathered is not None else 'UrbanSound8K')

            # Log script output to file
            full_save_path = join('saved_models', self.save_dir)
            os.makedirs(full_save_path)

            handler = logging.FileHandler(join(full_save_path, 'output.log'))
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)

        # Init data manager
        logging.info('Initializing data manager')
        if gathered is not None:
            # Use dataset gathered by scraper
            if os.environ.get('SOUNDSCRAPE_AUTH_JSON') is None:
                raise EnvironmentError('SOUNDSCRAPE_AUTH_JSON must be an environment variable containing path to service account JSON credentials file')
            self.dm = SoundSortDataManager(dataset_path, os.environ['SOUNDSCRAPE_AUTH_JSON'], 'soundscrape-bucket', gathered.split(','), train_class_pct=train_class_pct, file_duration=file_duration, sr=sr)
        else:
            # Use UrbanSound8K dataset
            self.dm = UrbanSoundDataManager(join(dataset_path, 'audio'), train_class_pct=train_class_pct, file_duration=file_duration, sr=sr)

        # Loss function used during training
        self.loss_function = nn.MSELoss()

        # Determine device for training
        self.use_cuda = (torch.cuda.device_count() > 1)
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
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, num_recurrent, self.dm.chunks, self.dm.chunk_len, label, self.device)
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

            if self.save:
                # Create model's save directory
                os.makedirs(join('saved_models', self.save_dir, str(label)))

    def test(self, model, save_path=None):
        '''
        Determine a model's accuracy against testing set
        '''
        total_test_files = len(self.dm.test_files)
        num_test_files = total_test_files - (total_test_files % self.batch_size)

        num_batches = num_test_files // self.batch_size
        abs_diff, false_pos, false_neg = 0, 0, 0
        for i in range(num_batches):
            # Get testing batch
            batch, labels = self.dm.get_batch('test', size=self.batch_size)
            batch.to(self.device)
            labels_tensor = torch.Tensor([float(label == model.label) for label in labels]).to(self.device)

            # Wipe state clean for next file (gross way to do it)
            try:
                model.module.init_state_tensors()
            except AttributeError:
                model.init_state_tensors()

            # Run model
            output = model(batch)

            # Free up memory
            del batch

            # Calculate accuracy
            output_rounded = torch.round(output.t())
            diff = (labels_tensor - output_rounded)

            for v in diff[0]:
                v = v.item()
                abs_diff += abs(v)
                false_pos += -1 * min(0, v)
                false_neg += max(0, v)

        if save_path is not None:
            # Save model
            torch.save(model.state_dict(), save_path)

        false_pos = float(false_pos)/abs_diff
        false_neg = float(false_neg)/abs_diff
        accuracy = 1 - (abs_diff / (self.batch_size*num_batches))

        return accuracy*100, false_pos*100, false_neg*100

    def train(self, epochs):
        logging.info('Begin training ({} epochs)'.format(epochs))

        batches_per_epoch = self.dm.num_train_files // self.batch_size

        # Train each model serially to minimize memory footprint
        for i, model in enumerate(self.models):
            # Move model to training device
            model.to(self.device)

            # Retrieve optimizer
            optimizer = self.optimizers[i]

            for e in range(epochs):
                for _ in range(batches_per_epoch):
                    model.zero_grad()

                    # Retrieve batch
                    batch, labels = self.dm.get_batch('train', size=self.batch_size, train_class=i)
                    batch.to(self.device)

                    # Wipe state clean for next file (gross way to do it)
                    try:
                        model.module.init_state_tensors()
                    except AttributeError:
                        model.init_state_tensors()

                    # Run network
                    output = model(batch)

                    # Free up memory
                    del batch

                    # Determine correct output for this batch
                    correct_output = torch.reshape(torch.Tensor([float(label == model.label) for label in labels]), (self.batch_size, 1)).to(self.device)

                    # Calculate loss
                    loss = self.loss_function(output, correct_output)

                    # Backpropagate
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if (e+1) % (epochs//10) == 0:
                    # Run against test set
                    if self.save:
                        # Save a snapshot
                        save_path = join('saved_models', self.save_dir, str(model.label), '{}_epochs.pth'.format(e+1))
                    else:
                        save_path = None

                    accuracy, false_pos, false_neg = self.test(model, save_path=save_path)
                    logging.info('({}/{}) model {}: accuracy = {:.2f}% ({:.2f}% false positive, {:.2f}% false negative)'.format(e+1, epochs, model.label, accuracy, false_pos, false_neg))

                    if accuracy >= self.min_accuracy*100:
                        # Model has met minimum criteria, exit now
                        break

            # Free up memory
            del model

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
        num_batches = num_test_files // self.batch_size

        # Record all output
        output = torch.zeros([len(self.dm.classes), num_test_files], dtype=torch.float32)
        labels = torch.zeros([num_test_files], dtype=torch.int16)

        # Load and test each model
        for i, c in enumerate(self.dm.classes):
            logging.info('Running model {} ({})'.format(i, c))

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
                    # Model (probably) trained with GPU and we're now trying to load it with a CPU
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[ k[7:] ] = v

                    model.load_state_dict(new_state_dict)

            # Collect all output for testing set
            model.to(self.device)
            model.eval()
            for j in range(num_batches):
                batch, batch_labels = self.dm.get_batch('test', size=self.batch_size)
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

                # Free up memory
                del batch

        # Calculate accuracy
        output = output.t()
        correct = 0
        total = 0
        for i in range(num_test_files):
            if labels[i].item() < len(self.dm.classes):
                total += 1
                correct += (output[i].argmax().item() == labels[i].item())

        logging.info('Correct: {}/{} ({}% accuracy)'.format(correct, total, float(correct)/total*100))

if __name__ == '__main__':
    # Create argument parser for easier CLI
    parser = argparse.ArgumentParser(description='Binary classifier models for UrbanSound classification',
        prog='classifier.py')
    parser.add_argument('-p', '--path', type=str, required=True,
        help='path to UrbanSound[8K] dataset')
    parser.add_argument('--hidden', type=int, required=True,
        help='dimension of hidden internal layers of network')
    parser.add_argument('-b', '--batch', type=int, required=True,
        help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.005,
        help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=True,
        help='number of training epochs')
    parser.add_argument('-t', '--target', type=float, default=0.5,
        help='fraction of training set to be of target class (accomplished by resampling target class to correct imbalance)')
    parser.add_argument('-d', '--duration', type=float, default=2,
        help='number of seconds used from each file in dataset (those shorter are zero-padded)')
    parser.add_argument('-r', '--recurrent', type=int, default=3,
        help='number of recurrent layers in each model')
    parser.add_argument('--sr', type=int, default=16000,
        help='sample rate at which files are loaded')
    parser.add_argument('-a', '--accuracy', type=float, default=0.9,
        help='stop training a model as soon as it reaches this accuracy score')
    parser.add_argument('-s', '--save', action='store_true', default=False,
        help='flag to save network and log output')
    parser.add_argument('-c', '--card', type=int, default=None,
        help='index of CUDA-capable GPU to be used for training and testing')
    parser.add_argument('-g', '--gathered', type=str, default=None,
        help='comma-separated classes to identify within dataset gathered by soundScrape')
    parser.add_argument('-l', '--load', type=str, default=None,
        help='load and test entire classifier against test set')
    args = parser.parse_args()

    # Set log level to info
    logging.getLogger().setLevel(logging.INFO)

    # Create and train classifier
    classifier = Classifier(args.path, args.hidden, args.batch, args.recurrent,args.lr, args.sr, args.duration, args.card, args.target,
        args.accuracy, args.save, args.gathered)

    if args.load is not None:
        # Test entire system by running test set through every classifier
        classifier.full_test(saved_classifiers_path=args.load)
    else:
        # Train all classifiers to identify their respective sounds
        classifier.train(args.epochs)
        classifier.full_test()
