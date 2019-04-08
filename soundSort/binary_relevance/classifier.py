import torch
import torch.nn as nn
import torch.optim as optim
import logging

from UrbanSoundDataManager import UrbanSoundDataManager
from sys import argv, stderr

# Model params
hidden_dim = 100
batch_dim = 100
lr = 0.08
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
            self.h = torch.randn(1, self.batch_size, self.hidden_size).to(self.device)
            self.c = torch.randn(1, self.batch_size, self.hidden_size).to(self.device)

        def __init__(self, input_size, hidden_size, batch_size, chunks, chunk_len, label, device):
            super(Classifier.Model, self).__init__()

            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.chunks = chunks
            self.chunk_len = chunk_len
            self.label = label
            self.device = device

            # Define model
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear_portion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
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
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, self.dm.chunks, self.dm.chunk_len, label, self.device)
            model.float()

            if self.use_cuda:
                # Prep model to train on GPU
                label = model.label
                model = nn.DataParallel(model, device_ids=self.device_ids)

                # Hack to get DataParallel object to maintain Model object's label
                setattr(model, 'label', label)

            self.models.append(model)

            # Optimizers specific to each model
            self.optimizers.append(optim.SGD(model.parameters(), lr=lr))

    def test(self, model):
        '''
        Determine a model's accuracy against testing set
        '''
        total_test_files = len(self.dm.test_files)
        num_test_files = total_test_files - (total_test_files % self.batch_size)

        c_true, o_true = 0, 0
        num_batches = num_test_files//self.batch_size
        for i in range(num_batches):
            # Get testing batch
            batch, labels = self.dm.get_batch('test', size=self.batch_size)
            batch.to(self.device)
            labels = torch.Tensor([float(label == model.label) for label in labels])

            # Run model
            output = model(batch)

            # Get number of mislabeled files
            c_true += torch.sum(labels).item()
            o_true += torch.sum(torch.round(output)).item()
            for i in range(len(labels)):
                if labels[i]:
                    logging.info(output[i][0].item())

        return c_true, o_true

    def train(self, epochs):
        logging.info('Begin training')

        for i in range(len(self.models)):
            # Retrieve model
            model = self.models[i]
            model.to(self.device)

            for e in range(epochs):
                # Retrieve batch
                batch, labels = self.dm.get_batch('train', size=self.batch_size)
                batch.to(self.device)

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
                    c_true, o_true = self.test(model)
                    logging.info('({}/{}) model {}: c_true = {} o_true = {}'.format(e+1, epochs, i, c_true, o_true))
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
