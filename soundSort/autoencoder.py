import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

from SoundSortDataManager import SoundSortDataManager as ssdm

logging.getLogger().setLevel(logging.INFO)

# High-level params
sample_rate = 1000
duration = 10
num_epochs = 100
alpha = 0.001
layer_dims = [sample_rate, sample_rate-200, sample_rate-400]

class AutoEncoder(nn.Module):
    def __init__(self, layer_dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        iters = len(layer_dims) - 1
        for i in range(iters):
            self.encoder.add_module('encoder_{}'.format(i),
                nn.Linear(layer_dims[i], layer_dims[i+1]))


            i_rev = iters - i
            self.decoder.add_module('decoder_{}'.format(i),
                nn.Linear(layer_dims[i_rev], layer_dims[i_rev-1]))

    def forward(self, x):
        return self.decoder(self.encoder(x))

dm = ssdm('soundScrapeDumps', '../soundScrape-d78c4b542d68.json',
    'soundscrape-bucket', sample_rate=sample_rate, duration=duration)
dm.prep_data(file_name='traffic-street_93a4d47b50b5f221326d17489dfcb0e87f62b8f1.wav')

# Create network
auto_encoder = AutoEncoder(layer_dims).float()
cross_entropy = torch.nn.L1Loss()
optimizer = torch.optim.Rprop(auto_encoder.parameters(), lr=alpha)

# Train network
for epoch in range(1, num_epochs):
    for _ in range(duration):
        chunk = dm.next_chunk()
        fft = np.fft.fft(chunk).real.astype(np.float)
        fft_tensor = torch.from_numpy(fft).float()
        output = auto_encoder(fft_tensor)
        loss = cross_entropy(output, fft_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _ == duration-1:
            logging.info('{}/{} loss = {}'.format(epoch, num_epochs, loss))
