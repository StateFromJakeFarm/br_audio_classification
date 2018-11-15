import os
import librosa
import logging
import numpy as np

from google.cloud import storage
from sys import stderr
from pydub import AudioSegment
from random import shuffle

class SoundSortDataManager(object):
    '''
    Download and prepare files from Google Cloud Storage bucket given a
    service account's JSON credentials file.
    '''
    def __init__(self, data_dir, auth_json, bucket, rnn=False):
        self.data_dir = data_dir
        self.data_x = []
        self.data_y = []
        self.i = 0
        self.rnn = rnn

        # Get authorization JSON
        if os.path.isfile(auth_json):
            try:
                client = storage.Client.from_service_account_json(auth_json)
                self.bucket = client.get_bucket(bucket)
            except:
                logging.error('Could not login to service account')
                exit(1)
        else:
            logging.error('{} does not exist'.format(auth_json))
            exit(1)

        # Create storage directory if it does not exist
        if not os.path.isdir(self.data_dir):
            try:
                logging.info('Creating {}'.format(self.data_dir))
                os.mkdir(self.data_dir)
            except:
                logging.error('Could not create {}'.format(self.data_dir))
                exit(1)

    def list(self):
        '''
        List bucket contents (returns iterator).
        '''
        return self.bucket.list_blobs()

    def download(self, gcs_path, local_path):
        '''
        Download a file from bucket to local disk.
        '''
        logging.info('Saving {} on from GCS bucket {} to {} locally'.format(gcs_path, self.bucket.name, local_path))

        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

    def convert_wav(self, src_path, dest_path):
        '''
        Convert file at src_path to WAV format and save converted file at
        dest_path.
        '''
        logging.info('Converting {} to {}'.format(src_path, dest_path))

        ext = src_path.split('.')[-1].strip()
        {
            'mp3': AudioSegment.from_mp3,
            'ogg': AudioSegment.from_ogg,
            'flv': AudioSegment.from_flv,
        }.get(ext)(src_path).export(dest_path, 'wav')

    def extract_features(self, file_path, cepstra=26):
        '''
        Extract features from file.
        '''
        try:
            Y, sr = librosa.load(file_path)
            if self.rnn:
                return librosa.feature.mfcc(y=Y, sr=sr, n_mfcc=cepstra)
            else:
                stft = np.abs(librosa.stft(Y))
                mfccs = np.mean(librosa.feature.mfcc(y=Y, sr=sr, n_mfcc=40).T,axis=0)
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
                mel = np.mean(librosa.feature.melspectrogram(Y, sr=sr).T,axis=0)
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(Y), sr=sr).T,axis=0)

                return np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
        except Exception as e:
            logging.warning('Failed to extract features from {}: {}'.format(file_path, e))
            return np.empty(0)

    def prep_data(self, num_timesteps=100):
        '''
        Extract features from all files so they can be easily sliced into batches
        during training.
        '''
        # Determine which files within the bucket we already have downloaded and
        # which we need to download now
        logging.info('Getting list of local files')
        local_files  = os.listdir(self.data_dir)
        logging.info('Getting list of remote files')
        remote_blobs = self.list()
        for blob in remote_blobs:
            remote_name_no_ext = blob.name.split('.')[0]
            if remote_name_no_ext + '.wav' not in local_files:
                # We don't have this file (or a WAV version of it)
                local_path = self.data_dir + '/' + blob.name
                dest_path  = self.data_dir + '/' + remote_name_no_ext + '.wav'

                # Download from GCP storage
                logging.info('Downloading {}'.format(blob.name))
                self.download(blob.name, local_path)

                if local_path.split('.')[-1] != 'wav':
                    # File is not in WAV format, convert and delete original
                    logging.info('Converting {} to WAV format'.format(blob.name))

                    self.convert_wav(local_path, dest_path)
                    os.remove(local_path)

        # Perform feature extraction on all files
        data = []
        for file_path in os.listdir(self.data_dir):
            file_path = '{}/{}'.format(self.data_dir, file_path)
            logging.info('Extracting features from {}'.format(file_path))

            features = self.extract_features(file_path)
            if features.shape[0] == 0:
                # Could not extract features
                logging.warning('Failed to extract features from {}'.format(file_path))
                continue

            if self.rnn:
                # Reshape data to be time-step major
                ceptra = features.shape[0]
                time_slots = features.shape[1]

                features = features.reshape((time_slots, ceptra))

                if time_slots < num_timesteps:
                    # Needs padding to reach num_timesteps
                    features = np.append(features, np.zeros((num_timesteps-time_slots, ceptra)))
                    features = features.reshape((num_timesteps, ceptra))
                elif time_slots > num_timesteps:
                    # Needs truncating
                    features = features[:num_timesteps]
            else:
                # Only flatten for regular feed-forward
                features = features.flatten()

            # Get search terms associated with this file by soundScrape
            matched_terms = os.path.split(file_path)[-1].split('_')[0].split('-')

            # Add row to dataset
            data.append([features, [(1 if 'saw' in matched_terms else 0)]])

        # Shuffle rows
        shuffle(data)
        for row in data:
            self.data_x.append(row[0])
            self.data_y.append(row[1])

    def next_batch(self, batch_size=50):
        '''
        Return numpy array of next training batch.
        '''
        batch_x = []
        batch_y = []
        while len(batch_x) < batch_size:
            batch_x.append(self.data_x[self.i])
            batch_y.append(self.data_y[self.i])

            if self.i >= len(batch_x):
                self.i = 0

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y
