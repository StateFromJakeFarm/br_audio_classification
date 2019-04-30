import os
import librosa
import logging
import numpy as np

from google.cloud import storage
from sys import stderr
from pydub import AudioSegment
from random import shuffle

class SoundSortDataManager():
    '''
    Download and prepare files from Google Cloud Storage bucket given a
    service account's JSON credentials file.
    '''
    def __init__(self, data_dir, auth_json_path, bucket_name, sr=16000, file_duration=4, chunk_duration=0.1, train_class_pct=0.5):
        # Local directory for all audio files
        self.data_dir = data_dir

        # Sample rate at which to load files
        self.sample_rate = sample_rate

        # Max seconds of audio to load for each file
        self.duration = duration

        # List to store NumPy arrays containing sound data
        self.data = []

        # Keep track of terms associated with each file
        self.file_terms = []

        self.i = 0 # Current file
        self.j = 0 # Next sample

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
        logging.info('Saving {} from GCS bucket {} to {} locally'.format(gcs_path, self.bucket.name, local_path))

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

    def load_file(self, file_path, duration):
        '''
        Load a sound file
        '''
        try:
            self.file_terms.append(set(file_path.split('_')[0].split('-')))
            file_path = os.path.join(self.data_dir, file_path)
            logging.info('loading {}'.format(file_path))

            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            self.data.append(y)
        except Exception as e:
            logging.error('Failed to load {}: {}'.format(file_path, e))

    def prep_data(self, num_files=None, file_name=None):
        '''
        Load training files
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

        # Load sound data into list of np.array objects
        if file_name:
            # If user wants one specific file, only load that one
            self.load_file(file_name, self.duration)
        else:
            for i, file_path in enumerate(os.listdir(self.data_dir)):
                if num_files and i >= num_files:
                    # We've prepped all the files the user wanted
                    break

                self.load_file(file_path, self.duration)

    def next_file(self):
        '''
        Move to next file in dataset
        '''
        self.i += 1
        if self.i >= len(self.data):
            self.i = 0

    def next_chunk(self, sec=1):
        '''
        Serve next chunk of samples
        '''
        chunk_size = self.sample_rate * sec
        chunk = self.data[self.i][self.j:self.j+chunk_size]
        self.j += chunk_size

        if (self.j+chunk_size)/self.sample_rate > self.duration:
            self.j = 0

        return chunk

    def get_file_terms(self):
        '''
        Return set of terms identified by scraper when file was discovered
        '''
        return self.file_terms[self.i]

    def next_chunk_and_label(self, target_term):
        '''
        Return tuple of (chunk, 1/0) where the second element is a boolean
        denoting whether or not the file's description contains the
        target term
        '''
        has_term = int(1 if target_term in get_file_terms() else 0)
        return (self.next_chunk(), np.array([has_term]))
