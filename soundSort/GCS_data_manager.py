import os
import logging

from google.cloud import storage
from sys import stderr
from pydub import AudioSegment

class GCS_data_manager(object):
    '''
    Download and prepare files from Google Cloud Storage bucket given a
    service account's JSON credentials file.
    '''
    def __init__(self, auth_json, bucket):
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
