import os
import librosa
import logging
import numpy as np
import tensorflow as tf

from SoundSort_data_manager import SoundSort_data_manager

# Configure high-level params
data_dir = 'soundScrapeDumps'
auth_json_path = '../soundScrape-d78c4b542d68.json'
bucket_name = 'soundscrape-bucket'
logging.basicConfig(level=logging.INFO)

# Configure model
num_timesteps = 200
steps = 1000
batch_size = 20
cepstra = 26
hidden_layer_size = 50

# Prep data
dm = SoundSort_data_manager(data_dir, auth_json_path, bucket_name, rnn=True)
dm.prep_data(num_timesteps=num_timesteps)

# Create model
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)

inputs = tf.placeholder(tf.float32, shape=[None, num_timesteps, cepstra], name='input_layer')
labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)
last_rnn_output = rnn_outputs[:,-1,:]

W_1 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], mean=0, stddev=0.01), name='fully-connected_1')
b_1 = tf.Variable(tf.truncated_normal([hidden_layer_size], mean=0, stddev=0.01), name='bias_1')
h_1 = tf.nn.relu(tf.matmul(last_rnn_output, W_1) + b_1)

W_2 = tf.Variable(tf.truncated_normal([hidden_layer_size, 1], mean=0, stddev=0.01), name='fully-connected_2')
final_outputs = tf.matmul(h_1, W_2)

cross_entropy = tf.abs(final_outputs - labels)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

prediction = tf.cast(final_outputs, tf.float32)

init = tf.global_variables_initializer()

# Run model
with tf.Session() as sess:
    sess.run(init)

    for step in range(steps):
        # Train on next batch
        train_data, train_labels = dm.next_batch(batch_size=batch_size)
        sess.run(train_step, feed_dict={inputs: train_data, labels: train_labels})

        if step % (steps/10) == 0:
            # Test on next item in training set (not genuine testing set...)
            test_data, test_label = dm.next_batch(batch_size=1)
            pred = sess.run(prediction, feed_dict={inputs: test_data, labels: test_label})
            logging.info('({}/{})  predicted: {}  actual: {}'.format(step, steps, pred, test_label))
