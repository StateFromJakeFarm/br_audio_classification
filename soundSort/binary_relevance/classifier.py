import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from UrbanSoundDataManager import UrbanSoundDataManager
from sys import argv, stderr

logging.getLogger().setLevel(logging.INFO)

# Model params
hidden_dim = 800
batch_dim = 100
lr = 0.005

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

    def __init__(self, audio_dir, hidden_size, batch_size, lr=0.005):
        self.dm = UrbanSoundDataManager(audio_dir)
        self.batch_size = batch_size

        # Loss function used during training
        self.loss_function = nn.L1Loss()

        self.models = []
        self.optimizers = []
        for label in range(1, 11):
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, self.dm.chunks, self.dm.chunk_len, label)
            model.float()
            self.models.append(model)

            # Optimizers specific to each model
            self.optimizers.append(optim.SGD(model.parameters(), lr=lr))

    def train(self, epochs):
        for e in range(epochs):
            batch, labels = self.dm.get_batch('train', size=self.batch_size)

            for i in range(len(self.models)):
                model = self.models[i]
                optimizer = self.optimizers[i]

                # Run network
                output = model(batch)

                # Determine correct output for this batch
                correct_output = torch.Tensor([float(label == model.label) for label in labels])

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

if len(argv) == 2:
    classifier = Classifier(argv[1], hidden_dim, batch_dim, lr)
    classifier.train(50)
else:
    print('USAGE: classifier.py <path to audio dir>', file=stderr)
    exit(1)
