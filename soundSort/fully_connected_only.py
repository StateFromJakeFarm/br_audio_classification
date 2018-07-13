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

logging.basicConfig(level=logging.INFO)

def load_sound_files(file_paths):
    raw_sounds = []
    for path in file_paths:
        x, sr = librosa.load(path)
        raw_sounds.append(x)

    return raw_sounds

dm = GCS_data_manager('../soundScrape-d78c4b542d68.json', 'soundscrape-bucket')
