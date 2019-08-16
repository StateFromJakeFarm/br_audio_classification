import os
import torch
import librosa
import logging
import numpy as np

from google.cloud import storage
from sys import stderr
from pydub import AudioSegment
from random import shuffle

class SoundSortDataManager:
    '''
    Download and prepare files from Google Cloud Storage bucket given a
    service account's JSON credentials file.
    '''
    def __init__(self, data_dir_path, auth_json_path, bucket_name, classes, batch_size=10, sr=8000, file_duration=4, chunk_duration=0.1, train_class_pct=0.5):
        # Local directory for all audio files
        self.data_dir_path = data_dir_path

        # Max seconds of audio to load for each file
        self.file_duration = file_duration

        # Sample rate
        self.sr = sr

        # Classes for each sample
        self.classes = classes
        logging.info('Classes: {}'.format(str(classes)))

        self.batch_size = batch_size

        # Keep track of terms associated with each file
        self.file_terms = []

        # Calculate file data dimensions
        self.file_duration = file_duration
        self.samples_per_file = self.sr*file_duration
        self.chunks = int(file_duration/chunk_duration)
        self.chunk_len = int(chunk_duration*self.sr)

        self.i = 0 # Current file
        self.j = 0 # Next sample

        # Get authorization JSON
        if os.path.isfile(auth_json_path):
            client = storage.Client.from_service_account_json(auth_json_path)
            self.bucket = client.get_bucket(bucket_name)
        else:
            logging.error('{} does not exist'.format(auth_json_path))
            exit(1)

        # Create storage directory if it does not exist
        if not os.path.isdir(self.data_dir_path):
            logging.debug('Creating {}'.format(self.data_dir_path))
            os.makedirs(self.data_dir_path)

        # Keep track of each file's class
        files = []
        self.file_classes = {}

        # Determine which files within the bucket we already have downloaded and
        # which we need to download now
        logging.debug('Getting list of local files')
        local_files  = os.listdir(self.data_dir_path)
        logging.debug('Getting list of remote files')
        for blob in self.list():
            remote_name_no_ext = blob.name.split('.')[0]
            dest_path = os.path.join(self.data_dir_path, remote_name_no_ext + '.wav')
            if remote_name_no_ext + '.wav' not in local_files:
                # We don't have this file (or a WAV version of it)
                local_path = os.path.join(self.data_dir_path, blob.name)

                # Download from GCP storage
                logging.debug('Downloading {}'.format(blob.name))
                self.download(blob.name, local_path)

                if local_path.split('.')[-1] != 'wav':
                    # File is not in WAV format, convert and delete original
                    logging.debug('Converting {} to WAV format'.format(blob.name))

                    self.convert_wav(local_path, dest_path)
                    os.remove(local_path)

            # Assign file to class based on search terms it contains (this is
            # done greedily, so if the file matches multiple classes, it will be
            # placed in the class of the first search term it matches)
            file_terms = self.get_terms(dest_path)
            for i, c in enumerate(self.classes + ['other']):
                if c in file_terms or c == 'other':
                    self.file_classes[dest_path] = i
                    break

            files.append(dest_path)

        # Designate 75% of files for training and the rest for testing
        shuffle(files)
        self.num_train_files = int(len(files)*0.75)
        train_files = files[:self.num_train_files]
        self.test_files = files[self.num_train_files:]

        # Count how many occurrences of each class are in training set
        train_set_class_counts = [0 for c in self.classes + ['other']]
        for f in train_files:
            train_set_class_counts[self.file_classes[f]] += 1

        # (Likely) Overselect the class being trained for on each model because
        # dataset is very unbalanced
        self.train_files_by_class = []
        for c in range(len(self.classes)):
            # True files must account for train_class_pct of training set
            p_true = train_class_pct/train_set_class_counts[c]
            # False files must account for (1 - train_class_pct) of training set
            p_false = (1 - train_class_pct)/(self.num_train_files - train_set_class_counts[c])

            p = [p_true if self.file_classes[f] == c else p_false for f in train_files]

            # Select training set
            self.train_files_by_class.append(np.random.choice(train_files, size=self.num_train_files,
                replace=True, p=p))

        # Iterators to keep track of where we are in the training and testing sets
        self.i_train = 0
        self.i_test = 0

        # Store training batches for current class being trained only (because
        # they are trained serially, and so only one class' training set needs
        # to be in memory at a time)
        self.training_batches = []

        # Store testing batches in memory
        logging.info('Loading testing batches into memory')
        self.testing_batches = []
        for batch in range(len(self.test_files) // self.batch_size):
            self.testing_batches.append(
                self.build_batch())

    def list(self):
        '''
        List bucket contents (returns iterator).
        '''
        return self.bucket.list_blobs()

    def download(self, gcs_path, local_path):
        '''
        Download a file from bucket to local disk.
        '''
        logging.debug('Saving {} from GCS bucket {} to {} locally'.format(gcs_path, self.bucket.name, local_path))

        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

    def convert_wav(self, src_path, dest_path):
        '''
        Convert file at src_path to WAV format and save converted file at
        dest_path.
        '''
        logging.debug('Converting {} to {}'.format(src_path, dest_path))

        ext = src_path.split('.')[-1].strip()
        {
            'mp3': AudioSegment.from_mp3,
            'ogg': AudioSegment.from_ogg,
            'flv': AudioSegment.from_flv,
        }.get(ext)(src_path).export(dest_path, 'wav')

    def load_training_batches(self, current_training_class):
        '''
        Load training set for current training class into memory
        '''
        logging.info('Loading training batches into memory')
        self.training_batches = []
        for batch in range(len(self.train_files_by_class[current_training_class]) // self.batch_size):
            self.training_batches.append(
                self.build_batch(train_class=current_training_class))

    def build_batch(self, train_class=None, use_fft=False):
        '''
        Build next batch of shape (batch, seq_len, seq), which is representative
        of (file, chunks, chunk_len)
        '''
        # Gross hack to actually increment the iterator for the set being used
        if train_class is not None and 0 < train_class and train_class < len(self.classes):
            file_set = self.train_files_by_class[train_class]
            iterator = self.i_train

            # Increment iterator
            self.i_train += self.batch_size
            if self.i_train + self.batch_size >= len(self.train_files_by_class[train_class]):
                self.i_train = 0
        else:
            file_set = self.test_files
            iterator = self.i_test

            # Increment iterator
            self.i_test += self.batch_size
            if self.i_test + self.batch_size >= len(self.test_files):
                self.i_test = 0

        # Compile file data for this chunk into tensor
        batch = np.zeros((self.batch_size, self.chunks, self.chunk_len), dtype=float)
        labels = []
        for i, file in enumerate(file_set[iterator:iterator+self.batch_size]):
            # Extract label
            labels.append(self.file_classes[file])

            # Load data
            Y, sr = librosa.core.load(file, sr=self.sr, duration=self.file_duration)

            if Y.shape[0] < self.samples_per_file:
                # Pad this array with zeros on the end
                Y = np.pad(Y, (0, int(self.samples_per_file-Y.shape[0])), 'constant')

            # Chunk-up data
            for chunk in range(self.chunks):
                batch[i][chunk] = Y[chunk*self.chunk_len:chunk*self.chunk_len+self.chunk_len]

                if use_fft:
                    # Use FFT of chunk
                    batch[i][chunk] = fft(batch[i][chunk]).real

        return torch.from_numpy(batch.astype(np.float32)), labels

    def get_terms(self, file_path):
        '''
        Return terms associated with file
        '''
        return set(os.path.split(file_path)[1].split('_')[0].split('-'))
