import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from scipy.io.wavfile import write
from SoundSortDataManager import SoundSortDataManager as ssdm

logging.getLogger().setLevel(logging.INFO)

# High-level params
sample_rate = 4000
batch_size = 1000
num_batches = sample_rate / batch_size
duration = 2
num_epochs = 100
input_dim = batch_size
hidden_dim = int(batch_size * 1.25)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_cuda):
        super(AutoEncoder, self).__init__()
        self.lstm1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, input_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.h_t = torch.zeros(1, hidden_dim, dtype=torch.float)
        self.c_t = torch.zeros(1, hidden_dim, dtype=torch.float)
        self.h2_t = torch.zeros(1, hidden_dim, dtype=torch.float)
        self.c2_t = torch.zeros(1, hidden_dim, dtype=torch.float)

        if use_cuda:
            self.h_t = self.h_t.cuda()
            self.c_t = self.c_t.cuda()
            self.h2_t = self.h2_t.cuda()
            self.c2_t = self.c2_t.cuda()

    def forward(self, input_data):
        self.h_t, self.c_t = self.lstm1(input_data, (self.h_t, self.c_t))
        self.h2_t, self.c2_t = self.lstm2(self.h_t, (self.h2_t, self.c2_t))
        output = self.linear(self.h2_t)

        return output


dm = ssdm('soundScrapeDumps', '../soundScrape-d78c4b542d68.json',
    'soundscrape-bucket', sample_rate=sample_rate, duration=duration)
dm.prep_data(file_name='wind-metal-wind-metal_69699137754365d68f61c826163d29e16ec7c901.wav')

# Create network
use_cuda = (torch.cuda.device_count() > 1)
auto_encoder = AutoEncoder(input_dim, hidden_dim, use_cuda)
loss_function = nn.L1Loss()
optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001)

device = torch.device('cpu')
if use_cuda:
    auto_encoder = torch.nn.DataParallel(auto_encoder, device_ids=[0])
    device = torch.device('cuda:0')

auto_encoder.to(device)
auto_encoder.float()

# Train network
for epoch in range(num_epochs):
    for sec in range(duration):
        chunk = dm.next_chunk()

        for batch in np.split(chunk, num_batches):
            fft = np.fft.fft(batch).real.astype(np.float)
            fft_tensor = torch.from_numpy(fft).float().unsqueeze(0)

            if use_cuda:
                fft_tensor = fft_tensor.cuda()

            auto_encoder.zero_grad()

            # Run network
            output = auto_encoder(fft_tensor)

            # Calculate loss
            loss = loss_function(output, fft_tensor)

            # Backpropagate
            loss.backward(retain_graph=True)
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
        fft_tensor = torch.from_numpy(fft).float().unsqueeze(0)

        output = auto_encoder(fft_tensor)
        if use_cuda:
            output = output.cpu()
        output = output.squeeze(0).detach().numpy()

        wav_data = np.concatenate((wav_data, np.fft.ifft(output).real.astype(np.float)))

write('autoencoder_output.wav', sample_rate, wav_data)
