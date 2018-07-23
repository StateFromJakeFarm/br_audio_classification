import glob
import os
import logging
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib.pyplot import specgram
from SoundSort_data_manager import SoundSort_data_manager

# Configuration
data_dir = 'soundScrapeDumps'
logging.basicConfig(level=logging.INFO)

def get_sound_files(dm, storage_dir_path):
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

# Prep all data
sound_file_paths = get_sound_files(dm, 'soundScrapeDumps')
dm.prep_data(sound_file_paths)

# Helper functions
def variable(shape):
    return tf.Variable(tf.random_normal(shape, mean=0, stddev=0.01))

# Create model
input_dim = 193
hidden_dim = 300

x = tf.placeholder(tf.float32, [None, input_dim])
y_ = tf.placeholder(tf.float32, [None, 1])

W_1 = variable([input_dim, hidden_dim])
b_1 = variable([hidden_dim])
h_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)

W_2 = variable([hidden_dim, hidden_dim])
b_2 = variable([hidden_dim])
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W_3 = variable([hidden_dim, 1])
y = tf.matmul(h_2, W_3)

# Define graph actions
cross_entropy = tf.abs(y - y_)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
prediction = tf.cast(y, tf.float32)

# Run model:
steps = 1000
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for step in range(steps):
        train_x, train_y = dm.next_batch()
        sess.run(train_step, feed_dict={x: train_x, y_: train_y})

        if step % 100 == 0:
            test_x, test_y = dm.next_batch(1)
            pred = sess.run(prediction, feed_dict={x: test_x, y_: test_y})
            logging.info('{}/{}: pred = {}  actual = {}'.format(step, steps, pred, test_y))
