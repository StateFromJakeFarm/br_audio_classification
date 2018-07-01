import os

from google.cloud import storage
from sys import stderr

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
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
