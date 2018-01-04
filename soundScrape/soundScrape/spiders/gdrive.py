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
        self.folder_id = '18UqSr7-b4wtrQExDhVGswu6a_6Xcz5DS'
        gauth = GoogleAuth()
        self.drive = GoogleDrive(gauth)

    def upload_file(self, local_path):
        '''Upload a file to location within our drive folder'''
        folder, fname = os.path.split(local_path)
        f = self.drive.CreateFile({
            'title': fname,
            'parents': [{'kind': 'drive#fileLink', 'id': self.folder_id}]
        })
        f.SetContentFile(local_path)
        f.Upload()
