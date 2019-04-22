import torch
import torch.nn as nn
import torch.optim as optim
import logging

from UrbanSoundDataManager import UrbanSoundDataManager
from sys import argv, stderr, getsizeof

# Model params
hidden_dim = 256
batch_dim = 256
lr = 0.005
epochs = 1000
train_class_pct = 0.5
file_duration = 4
num_rec_layers = 2
sr = 16000

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
            self.h = torch.randn(num_rec_layers, self.batch_size, self.hidden_size).to(self.device)
            self.c = torch.randn(num_rec_layers, self.batch_size, self.hidden_size).to(self.device)

        def __init__(self, input_size, hidden_size, batch_size, chunks, chunk_len, label, device):
            super(Classifier.Model, self).__init__()

            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.chunks = chunks
            self.chunk_len = chunk_len
            self.label = label
            self.device = device

            # Define model
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_rec_layers, batch_first=True)
            self.linear_portion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

            # Init hidden and cell states
            self.init_state_tensors()

        def forward(self, x):
            # Run network
            x, state_tuple = self.lstm(x, (self.h, self.c))
            self.h, self.c = state_tuple
            x = self.linear_portion.forward(x[:, -1, :])

            return x

    def __init__(self, audio_dir, hidden_size, batch_size, lr=0.005, device_id=None, train_class_pct=0.5):
        logging.info('Initializing data manager')
        self.dm = UrbanSoundDataManager(audio_dir, train_class_pct=train_class_pct, file_duration=file_duration, sr=sr)
        self.batch_size = batch_size

        # Loss function used during training
        self.loss_function = nn.L1Loss()

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
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, self.dm.chunks, self.dm.chunk_len, label, self.device)
            model.float()

            if self.use_cuda:
                # Prep model to train on GPU
                label = model.label
                model = nn.DataParallel(model, device_ids=[device_id])

                # Hack to get DataParallel object to maintain Model object's label
                setattr(model, 'label', label)

            self.models.append(model)

            # Optimizers specific to each model
            self.optimizers.append(optim.SGD(model.parameters(), lr=lr))

    def test(self, model):
        '''
        Determine a model's accuracy against testing set
        '''
        model.test()
        total_test_files = len(self.dm.test_files)
        num_test_files = total_test_files - (total_test_files % self.batch_size)

        c_true, o_true = 0, 0
        num_batches = num_test_files//self.batch_size
        for i in range(num_batches):
            # Get testing batch
            batch, labels = self.dm.get_batch('test', size=self.batch_size)
            batch.to(self.device)
            labels_tensor = torch.Tensor([float(label == model.label) for label in labels])

            # Run model
            output = model(batch)

            # Get number of mislabeled files
            c_true += torch.sum(labels_tensor).item()
            o_true += torch.sum(torch.round(output)).item()
            for i in range(len(labels)):
                if labels[i]:
                    print(output[i][0].item())
                elif i % 10:
                    print('NOT: {}'.format(output[i][0].item()))

        return c_true, o_true

    def train(self, epochs):
        logging.info('Begin training')

        for i, model in enumerate(self.models):
            # Move model to training device
            model.to(self.device)

            # Retrieve optimizer
            optimizer = self.optimizers[i]

            for e in range(epochs):
                model.train()
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

                # Determine correct output for this batch
                correct_output = torch.Tensor([float(label == model.label) for label in labels]).to(self.device)

                # Calculate loss
                loss = self.loss_function(output, correct_output)

                # Backpropagate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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
            device_id = argv[2]
        else:
            device_id = None

        # Train classifier
        classifier = Classifier(argv[1], hidden_dim, batch_dim, lr=lr, device_id=device_id, train_class_pct=train_class_pct)
        classifier.train(epochs)
    else:
        print('USAGE: classifier.py <path to audio dir> [comma-separated GPU ids]', file=stderr)
        exit(1)
