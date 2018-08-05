import os
import librosa
import numpy as np
import tensorflow as tf

from SoundSort_data_manager import SoundSort_data_manager

# Configuration
data_dir = 'soundScrapeDumps'
num_timesteps = 100
logging.basicConfig(level=logging.INFO)

# Helper functions
def get_sound_files(storage_dir_path):
    storage_dir_path = storage_dir_path.strip('/')

    sound_file_paths = []
    for file_name in os.listdir(storage_dir_path):
        sound_file_paths.append(storage_dir_path + '/' + file_name.split('/')[-1])

    return sound_file_paths

def load_sound_files(file_paths):
    raw_sounds = []
    for path in file_paths:
        x, sr = librosa.load(path)
        raw_sounds.append(x)

    return raw_sounds

dm = SoundSort_data_manager('../soundScrape-d78c4b542d68.json', 'soundscrape-bucket')

# Create data directory if it does not exist
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# Download all the data files we need
cur_files = os.listdir(data_dir)
for blob in dm.list():
    f_name_no_ext = blob.name.split('.')[0]
    if f_name_no_ext + '.wav' not in cur_files:
        local_path = data_dir + '/' + blob.name
        dest_path = data_dir + '/' + f_name_no_ext + '.wav'
        dm.download(blob.name, local_path)
        dm.convert_wav(local_path, dest_path)
        os.remove(local_path)

# Prep data
sound_file_paths = get_sound_files('soundScrapeDumps')
dm.prep_data(sound_file_paths, rnn=True, num_timesteps=num_timesteps)

# Create model
