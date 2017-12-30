import gspread
import logging
from oauth2client.service_account import ServiceAccountCredentials

class sheet_obj():
    sheet = None
    def __init__(self, auth_json, sheet_name):
        '''Return a sheet object corresponding to sheet with provided name'''
        scope = ['https://spreadsheets.google.com/feeds']

        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(auth_json, scope)
            client = gspread.authorize(creds)
            self.sheet = client.open(sheet_name).sheet1
        except:
            logging.error('Unable to obtain reference to ' + sheet_name)

    def get_start_urls(self):
        '''Return generator for all base URLs in the spreadsheet'''
        for i, url in enumerate(self.sheet.col_values(1)):
            if url != '' and i > 0:
                yield url

sheet = sheet_obj('soundScrape-58c9b8c5fc20.json', 'soundScrape Dashboard')
for url in sheet.get_start_urls():
    print(url)
