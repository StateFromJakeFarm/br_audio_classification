import os
import librosa
import logging
import numpy as np
import tensorflow as tf

from SoundSort_data_manager import SoundSort_data_manager

# Configure high-level params
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

dm = SoundSort_data_manager('../soundScrape-d78c4b542d68.json', 'soundscrape-bucket', rnn=True)

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
dm.prep_data(sound_file_paths, num_timesteps=num_timesteps)

# Configure model
steps = 1000
batch_size = 20
cepstra = 26
hidden_layer_size = 50

# Create model
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)

inputs = tf.placeholder(tf.float32, shape=[None, num_timesteps, cepstra], name='input_layer')
labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, 1], mean=0, stddev=0.01), name='fully-connected')
last_rnn_output = outputs[:,-1,:]
final_outputs = tf.matmul(last_rnn_output, Wl)

cross_entropy = tf.abs(final_outputs - labels)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

prediction = tf.cast(final_outputs, tf.float32)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for step in range(steps):
        train_data, train_labels = dm.next_batch(batch_size=batch_size)
        sess.run(train_step, feed_dict={inputs: train_data, labels: train_labels})

        if step % (steps/10) == 0:
            test_data, test_label = dm.next_batch(batch_size=1)
            pred = sess.run(prediction, feed_dict={inputs: test_data, labels: test_label})
            logging.info('({}/{})  predicted: {}  actual: {}'.format(step, steps, pred, test_label))
