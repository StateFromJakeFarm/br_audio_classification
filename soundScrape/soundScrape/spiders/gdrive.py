import gspread
import logging
import re
import os
from oauth2client.service_account import ServiceAccountCredentials

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

            if url != '' and self.sheet.cell(i+1, 2).value in ['x', 'X']:
                urls.append(url)
                self.sheet.update_cell(i+1, 2, '')

        return urls

    def get_search_terms(self):
        '''Return list of all global search terms'''
        terms = []
        for i, term in enumerate(self.sheet.col_values(4)):
            if term != '' and i > 0:
                terms.append(term)

        return terms

    def get_accept_threshold(self):
        '''Return the accept threshold from Google Sheet or default if unable'''
        default = 0.3

        sheet_val = self.sheet.cell(2, 7).value
        if sheet_val == '' or not re.search('^[0-9]*$', sheet_val) or int(sheet_val) > 100:
            logging.warning('Invalid accept threshold: "' + sheet_val + '"; setting to ' + str(default))
            return default

        return int(sheet_val) / 100

    def get_max_page(self):
        '''Return the max page depth from Google Sheet or default if unable'''
        default = 10

        sheet_val = self.sheet.cell(3, 7).value
        if sheet_val == '' or not re.search('^[0-9]*$', sheet_val):
            logging.warning('Invalid max page: "' + sheet_val + '"; setting to ' + str(default))
            return default

        return int(sheet_val)
