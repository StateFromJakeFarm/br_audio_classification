import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

from scipy.io.wavfile import write
from SoundSortDataManager import SoundSortDataManager as ssdm

logging.getLogger().setLevel(logging.INFO)

# High-level params
sample_rate = 12000
batch_size = 3000
num_batches = sample_rate / batch_size
duration = 3
num_epochs = 500
alpha = 0.001
layer_dims = [batch_size, batch_size-200, batch_size-400]

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
dm.prep_data(file_name='metal-bolt-lock-metal-bolt-lock_7bb0b7659124099b91d3689c872d7934becba3ec.wav')

# Create network
auto_encoder = AutoEncoder(layer_dims).float()
cross_entropy = torch.nn.L1Loss()
optimizer = torch.optim.Rprop(auto_encoder.parameters(), lr=alpha)

device = torch.device('cpu')
use_cuda = (torch.cuda.device_count() > 1)
if use_cuda:
    auto_encoder = torch.nn.DataParallel(auto_encoder, device_ids=[0])
    device = torch.device('cuda:0')

auto_encoder.to(device)

# Train network
for epoch in range(num_epochs):
    for sec in range(duration):
        # Get FFT for next chunk
        chunk = dm.next_chunk()
        for batch in np.split(chunk, num_batches):
            fft = np.fft.fft(batch).real.astype(np.float)
            fft_tensor = torch.from_numpy(fft).float()
            fft_tensor = fft_tensor.to(device)

            # Run network
            output = auto_encoder(fft_tensor)

            # Calculate loss
            loss = cross_entropy(output, fft_tensor)

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if sec == duration-1:
            logging.info('{}/{} loss = {}'.format(epoch+1, num_epochs, loss))

# Write decompressed version back to wavfile
dm.i = 0 # Go back to beginning of file
wav_data = np.empty(1, dtype=np.float64)
for sec in range(duration):
    chunk = dm.next_chunk()
    for batch in np.split(chunk, num_batches):
        fft = np.fft.fft(batch).real.astype(np.float)
        fft_tensor = torch.from_numpy(fft).float()

        output = auto_encoder(fft_tensor)
        if use_cuda:
            output = output.cpu()
        output = output.detach().numpy()

        wav_data = np.concatenate((wav_data, np.fft.ifft(output).real.astype(np.float)))

write('autoencoder_output.wav', sample_rate, wav_data)
