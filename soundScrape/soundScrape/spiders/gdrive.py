import gspread
import logging
import re
import os
from oauth2client.service_account import ServiceAccountCredentials

class SheetObj:
    def __init__(self, auth_json, sheet_name):
        '''Return a sheet object corresponding to sheet with provided name'''
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']

        self.sheet = None
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(auth_json, scope)
            client = gspread.authorize(creds)
            self.sheet = client.open(sheet_name).sheet1
        except:
            logging.error('Unable to obtain reference to ' + sheet_name)
            return None

    def get_start_urls(self, url_col=1, x_row=2):
        '''Return list of all starting/base URLs'''
        urls = []
        for i, url in enumerate(self.sheet.col_values(url_col)):
            if i == 0:
                continue

            if url != '' and self.sheet.cell(i+1, x_row).value in ['x', 'X']:
                urls.append(url)
                self.sheet.update_cell(i+1, x_row, '')

        return urls

    def get_search_terms(self, col=4):
        '''Return list of all global search terms'''
        terms = []
        for i, term in enumerate(self.sheet.col_values(col)):
            if term != '' and i > 0:
                terms.append(term)

        return terms

    def get_avoid_terms(self, col=6):
        '''Return list of terms to avoid'''
        terms = []
        for i, term in enumerate(self.sheet.col_values(col)):
            if term != '' and i > 0:
                terms.append(term)

        return terms

    def get_accept_threshold(self, row=2, col=9):
        '''Return the accept threshold from Google Sheet or default if unable'''
        default = 0.3

        sheet_val = self.sheet.cell(row, col).value
        if sheet_val == '' or not re.search('^[0-9]*$', sheet_val) or int(sheet_val) > 100:
            logging.warning('Invalid accept threshold: "' + sheet_val + '"; setting to ' + str(default))
            return default

        return int(sheet_val) / 100

    def get_max_page(self, row=3, col=9):
        '''Return the max page depth from Google Sheet or default if unable'''
        default = 10

        sheet_val = self.sheet.cell(row, col).value
        if sheet_val == '' or not re.search('^[0-9]*$', sheet_val):
            logging.warning('Invalid max page: "' + sheet_val + '"; setting to ' + str(default))
            return default

        return int(sheet_val)
