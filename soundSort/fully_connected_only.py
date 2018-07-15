import glob
import os
import logging
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib.pyplot import specgram
from GCS_data_manager import GCS_data_manager

# Configuration
data_dir = 'soundScrapeDumps'
logging.basicConfig(level=logging.INFO)

def get_sound_files(dm, storage_dir_path):
    storage_dir_path = storage_dir_path.strip('/')

    sound_file_paths = []
    for blob in dm.list():
        sound_file_paths.append(storage_dir_path + '/' + blob.name.split('/')[-1])

    return sound_file_paths

def load_sound_files(file_paths):
    raw_sounds = []
    for path in file_paths:
        x, sr = librosa.load(path)
        raw_sounds.append(x)

    return raw_sounds

dm = GCS_data_manager('../soundScrape-d78c4b542d68.json', 'soundscrape-bucket')

# Download all the data files we need
cur_files = os.listdir(data_dir)
for blob in dm.list():
    f_name = blob.name.split('/')[-1]
    f_name_no_ext = f_name.split('.')[0]
    if f_name_no_ext + '.wav' not in cur_files:
        local_path = data_dir + '/' + f_name
        dm.download(blob.name, local_path)
        dm.convert_wav(local_path, f_name_no_ext + '.wav')
        os.remove(local_path)

def plot_specgram(raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip('meme',raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

sound_file_paths = get_sound_files(dm, 'soundScrapeDumps')
plot_specgram(load_sound_files([f_name.split('.')[0] + '.wav' for f_name in sound_file_paths[0:9]]))
