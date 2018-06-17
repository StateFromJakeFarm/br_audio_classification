import os

from google.cloud import storage
from sys import stderr

class DataManager(object):
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
                print(file=stderr, 'Error: could not login to service account')
                exit(1)
        else:
            print(file=stderr, 'Error: {} does not exist'.format(auth_json))
            exit(1)

    def list(self):
        '''
        List bucket contents (returns iterator).
        '''
        return self.bucket.list_blobs()

    def download(gcs_path, local_path):
        '''
        Download a file from bucket to local disk.
        '''
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
