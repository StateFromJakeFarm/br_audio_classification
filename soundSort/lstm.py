import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.nn import relu
from tensorflow.python.saved_model import tag_constants
from SoundSortDataManager import SoundSortDataManager

def variable_normal(shape, name):
    '''
    Initialize a TF variable of a user-defined shape using a truncated normal
    distribution
    '''
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=0.01), name=name)

def placeholder(shape, name):
    '''
    Initialize a TF placeholder of a user-defined shape
    '''
    return tf.placeholder(tf.float32, shape=shape, name=name)

# High-level paramaters
data_dir = 'soundScrapeDumps'
auth_json_path = '../soundScrape-d78c4b542d68.json'
bucket_name = 'soundscrape-bucket'
save_dir = 'lstm_save'
logging.getLogger().setLevel(logging.INFO)

# Configure model
num_timesteps = 200
epochs = 500
batch_size = 10
cepstra = 26
hidden_layer_size = 300

# Prep data
dm = SoundSortDataManager(data_dir, auth_json_path, bucket_name, ['saw', 'grinder', 'cut'], rnn=True)
dm.prep_data(num_timesteps=num_timesteps)

# LSTM layer
lstm_cell = rnn_cell.LSTMCell(hidden_layer_size, use_peepholes=True,
    state_is_tuple=True)
inputs = placeholder([None, num_timesteps, cepstra], 'inputs')
labels = placeholder([None, 1], 'labels')
lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

# FC layers
W_1 = variable_normal([hidden_layer_size, hidden_layer_size], 'W_1')
b_1 = variable_normal([hidden_layer_size], 'b_1')
h_1 = relu(tf.matmul(lstm_outputs[:,-1], W_1) + b_1)

W_2 = variable_normal([hidden_layer_size, hidden_layer_size], 'W_2')
b_2 = variable_normal([hidden_layer_size], 'b_2')
h_2 = relu(tf.matmul(h_1, W_2) + b_1)

W_3 = variable_normal([hidden_layer_size, 1], 'W_3')
b_3 = variable_normal([1], 'b_3')
final_outputs = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)

# Define training step
cross_entropy = tf.losses.mean_squared_error(labels, final_outputs)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# Define prediction
prediction = tf.cast(final_outputs, tf.float32)

if len(sys.argv) == 1:
    # Save model if no load directory specified
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)

# Train model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)

    if len(sys.argv) == 2:
        # Load saved model
        logging.info('Loading saved model at: {}'.format(sys.argv[1]))
        tf.saved_model.loader.load(sess, [tag_constants.TRAINING], sys.argv[1])

    if len(sys.argv) == 1:
        builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING],
            signature_def_map=None, assets_collection=None)

    for epoch in range(epochs):
        # Grab next batch
        train_data, train_labels = dm.next_batch(batch_size=batch_size)

        # Feed batch into model
        sess.run(train_step, feed_dict={inputs: train_data, labels: train_labels})

        if epoch % (epochs/10) == 0:
            # Run "test" on next element in training set
            test_data, test_label = dm.next_batch(batch_size=1)
            pred = sess.run(prediction, feed_dict={inputs: test_data, labels: test_label})
            logging.info('({}/{})  predicted: {}  actual: {}'.format(epoch, epochs, pred, test_label))

if len(sys.argv) == 1:
    # Save trained model
    builder.save()
