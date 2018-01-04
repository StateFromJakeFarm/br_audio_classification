import gspread
import logging
import re
import os
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class sheet_obj:
    def __init__(self, auth_json, sheet_name):
        '''Return a sheet object corresponding to sheet with provided name'''
        scope = ['https://spreadsheets.google.com/feeds']

        self.sheet = None
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(auth_json, scope)
            client = gspread.authorize(creds)
            self.sheet = client.open(sheet_name).sheet1
        except:
            logging.error('Unable to obtain reference to ' + sheet_name)
            return None

    def get_start_urls(self):
        '''Return list of all starting/base URLs'''
        urls = []
        for i, url in enumerate(self.sheet.col_values(1)):
            if i == 0:
                continue

            if url != '' and self.sheet.cell(i+1, 2).value != 'yes':
                urls.append(url)

        return urls

    def get_search_terms(self):
        '''Return list of all global search terms'''
        terms = []
        for i, term in enumerate(self.sheet.col_values(5)):
            if term != '' and i > 0:
                terms.append(term)

        return terms

class drive_obj:
    def __init__(self):
        '''Return a drive object corresponding to the folder with provided name'''
        self.root_id = '18UqSr7-b4wtrQExDhVGswu6a_6Xcz5DS'

        # Authorize our server
        self.gauth = GoogleAuth()
        self.gauth.LoadCredentialsFile('saved_creds.txt')
        if not self.gauth.credentials:
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            try:
                self.gauth.Refresh()
            except:
                self.gauth.LocalWebserverAuth()
        else:
            self.gauth.Authorize()

        self.gauth.SaveCredentialsFile('saved_creds.txt')

        self.drive = GoogleDrive(self.gauth)

    def upload_file(self, local_path, dest_folder):
        '''Upload a file to location within our drive folder'''
        # Get the ID of the destination folder
        folders = self.get_folders()
        dest_id = ''
        for folder in folders:
            cur_name = folder[0]
            cur_id   = folder[1]
            if dest_folder == cur_name:
                dest_id = cur_id
                break

        if dest_id == '':
            logging.error('Could not find /soundScrape/%s' % dest_folder)
            return 1

        folder, fname = os.path.split(local_path)
        f = self.drive.CreateFile({
            'title': fname,
            'parents': [{'kind': 'drive#fileLink', 'id': dest_id}]
        })
        f.SetContentFile(local_path)
        try:
            f.Upload()
            return 0
        except:
            return 1

    def create_folder(self, folder_name):
        '''Create a folder in the root folder for this project and return its ID'''
        f = self.drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'kind': 'drive#fileLink', 'id': self.root_id}]
        })
        f.Upload()

        return f['id']

    def get_folders(self):
        '''Return a list of all folders in the root directory'''
        file_list = self.drive.ListFile({
            'q': "'%s' in parents and trashed=false" % self.root_id
        }).GetList()

        folder_list = []
        for f in file_list:
            if f['mimeType'] == 'application/vnd.google-apps.folder':
                folder_list.append( (f['title'], f['id']) )
        return folder_list
