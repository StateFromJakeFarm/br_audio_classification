import gspread
import logging
import re
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
