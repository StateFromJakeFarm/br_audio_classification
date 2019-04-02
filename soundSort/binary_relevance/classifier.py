import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from UrbanSoundDataManager import UrbanSoundDataManager

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

            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear_portion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.ReLU()
            )

            self.init_state_tensors()

        def forward(self, x):
            # Run network
            x, self.h = self.lstm(x, (self.h, self.c))
            x = self.linear_portion.forward(x[:, -1, :])

            # Forget everything
            self.init_state_tensors()

            return x

    def __init__(self, audio_dir, hidden_size, batch_size):
        self.dm = UrbanSoundDataManager(audio_dir)
        self.batch_size = batch_size

        self.models = []
        for label in range(1, 11):
            model = self.Model(self.dm.chunk_len, hidden_size, batch_size, self.dm.chunks, self.dm.chunk_len, label)
            model.float()
            self.models.append(model)

    def train(self, epochs):
        for e in range(epochs):
            batch, labels = self.dm.get_batch()

            for model in self.models:
                output = model(batch)

classifier = Classifier('/home/jakeh/Downloads/UrbanSound8K/audio', 800, 10)
classifier.train(10)
