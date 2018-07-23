import os
import librosa
import logging
import numpy as np

from google.cloud import storage
from sys import stderr
from pydub import AudioSegment
from random import shuffle

class SoundSort_data_manager(object):
    '''
    Download and prepare files from Google Cloud Storage bucket given a
    service account's JSON credentials file.
    '''
    def __init__(self, auth_json, bucket):
        self.data_x = []
        self.data_y = []
        self.i = 0

        if os.path.isfile(auth_json):
            try:
                client = storage.Client.from_service_account_json(auth_json)
                self.bucket = client.get_bucket(bucket)
            except:
                print('Error: could not login to service account', file=stderr)
                exit(1)
        else:
            print('Error: {} does not exist'.format(auth_json), file=stderr)
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

    def extract_features(self, file_path):
        '''
        Extract features from file.
        '''
        try:
            X, sample_rate = librosa.load(file_path)
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

            return np.concatenate([mfccs, chroma, mel, contrast, tonnetz])
        except Exception as e:
            logging.warning('Failed to extract features from {}: {}'.format(file_path, e))
            return np.empty(0)

    def prep_data(self, sound_file_paths):
        '''
        Extract features from all files so they can be easily sliced into batches
        during training.
        '''
        data = []
        for file_path in sound_file_paths:
            logging.info('Extracting features from {}'.format(file_path))

            features = self.extract_features(file_path).flatten()
            if features.shape[0] == 0:
                # Could not extract features
                continue

            matched_terms = os.path.split(file_path)[-1].split('_')[0].split('-')
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

        return batch_x, batch_y
