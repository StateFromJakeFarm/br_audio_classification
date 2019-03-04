import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from SoundSortDataManager import SoundSortDataManager as ssdm

logging.getLogger().setLevel(logging.INFO)

# High-level params
sample_rate = 8000
batch_size = 4000
num_batches = sample_rate / batch_size
duration = 2
num_epochs = 200
input_dim = batch_size
hidden_dim = int(batch_size * 1.25)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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


    def forward(self, input_data):
        self.h_t, self.c_t = self.lstm1(input_data, (self.h_t, self.c_t))
        self.h2_t, self.c2_t = self.lstm2(self.h_t, (self.h2_t, self.c2_t))
        output = self.linear(self.h2_t)

        return output


dm = ssdm('soundScrapeDumps', '../soundScrape-d78c4b542d68.json',
    'soundscrape-bucket', sample_rate=sample_rate, duration=duration)
dm.prep_data(file_name='wind-metal-wind-metal_69699137754365d68f61c826163d29e16ec7c901.wav')

# Create network
auto_encoder = AutoEncoder(input_dim, hidden_dim)
auto_encoder.float()
loss_function = nn.MSELoss()
optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001)

# Train network
for epoch in range(num_epochs):
    for sec in range(duration):
        chunk = dm.next_chunk()

        for batch in np.split(chunk, num_batches):
            print(chunk)
            print(batch)
            fft = np.fft.fft(batch).real.astype(np.float)
            fft_tensor = torch.from_numpy(fft).float().unsqueeze(0)

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
