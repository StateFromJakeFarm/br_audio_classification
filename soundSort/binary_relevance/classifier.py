import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from UrbanSoundDataManager import UrbanSoundDataManager
from sys import argv, stderr

# Model params
hidden_dim = 800
batch_dim = 100
lr = 0.005
epochs = 50

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
            self.h = torch.randn(1, self.batch_size, self.hidden_size)
            self.c = torch.randn(1, self.batch_size, self.hidden_size)

        def __init__(self, input_size, hidden_size, batch_size, chunks, chunk_len, label):
            super(Classifier.Model, self).__init__()

            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.chunks = chunks
            self.chunk_len = chunk_len
            self.label = label

            # Define model
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear_portion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.ReLU()
            )

            # Init hidden and cell states
            self.init_state_tensors()

        def forward(self, x):
            # Run network
            x, self.h = self.lstm(x, (self.h, self.c))
            x = self.linear_portion.forward(x[:, -1, :])

            # Forget everything
            self.init_state_tensors()

            return x

    def __init__(self, audio_dir, hidden_size, batch_size, lr=0.005, device_ids=[]):
        logging.info('Initializing data manager')
        self.dm = UrbanSoundDataManager(audio_dir)
        self.batch_size = batch_size

        # Loss function used during training
        self.loss_function = nn.L1Loss()

        # Determine device for training
        self.device_ids = device_ids
        self.use_cuda = (torch.cuda.device_count() > 1)
        if self.use_cuda and len(self.device_ids):
            # Use GPU if available
            logging.info('Using CUDA GPU')
            self.device = torch.device('cuda:0')
        else:
            # Otherwise use CPU
            logging.info('Using CPU')
            self.use_cuda = False
            self.device = torch.device('cpu')

        logging.info('Constructing models')
        self.models = []
        self.optimizers = []
        for label in range(1, 11):
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, self.dm.chunks, self.dm.chunk_len, label)
            model.float()

            if self.use_cuda:
                # Prep model to train on GPU
                model = nn.DataParallel(model, device_ids=self.device_ids)

            self.models.append(model)

            # Optimizers specific to each model
            self.optimizers.append(optim.SGD(model.parameters(), lr=lr))

    def train(self, epochs):
        logging.info('Begin training')
        for e in range(epochs):
            batch, labels = self.dm.get_batch('train', size=self.batch_size)
            batch.to(self.device)

            for i in range(len(self.models)):
                # Retrieve model
                model = self.models[i]
                model.to(self.device)

                # Retrieve optimizer
                optimizer = self.optimizers[i]

                # Run network
                output = model(batch)

                # Determine correct output for this batch
                correct_output = torch.Tensor([float(label == model.label) for label in labels]).to(self.device)

                # Calculate loss
                loss = self.loss_function(output, correct_output)

                # Backpropagate
                loss.backward(retain_graph=True)
                optimizer.step()

                if (e+1) % (epochs/10) == 0:
                    # Run against test set
                    batch, labels = self.dm.get_batch('test', size=self.batch_size)
                    output = model(batch).detach().numpy()
                    correct_output = np.array([float(label == model.label) for label in labels])

                    logging.info('({}/{}) model {}: error = {}'.format(e+1, epochs, i+1, np.mean(np.abs(output - correct_output))))
        logging.info('Finish training')

if __name__ == '__main__':
    if len(argv) in [2, 3]:
        # Set log level
        logging.getLogger().setLevel(logging.INFO)

        # Determine if user selected GPUs to use for training
        if len(argv) == 3:
            device_ids = [int(i) for i in argv[2].split(',')]
        else:
            device_ids = []

        # Train classifier
        classifier = Classifier(argv[1], hidden_dim, batch_dim, lr=lr, device_ids=device_ids)
        classifier.train(epochs)
    else:
        print('USAGE: classifier.py <path to audio dir> [comma-separated GPU ids]', file=stderr)
        exit(1)
