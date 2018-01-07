import scrapy
import logging
from soundScrape.items import SoundFile
from .helper import *
from .gdrive import sheet_obj
from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider

class SoundSpider(CrawlSpider):
    '''Scrape sites for sound files'''
    # Name of spider
    name = "sound"

    # Where we start (from sheet)
    start_urls = []
    # Where we've been (from sheet)
    pages_visited = []

    # Terms each site will be searched for (from sheet)
    search_terms = []

    # Sound file extensions we'll look for
    sound_file_types = [
        'mp3'
    ]

    base_url_types = [
        '.com',
        '.org',
        '.gov',
        '.co.uk',
        '.edu'
    ]

    # Terms we'll use to identify "next page" links
    next_page_terms = [
        'page',
        'next page',
        '?p'
    ]

    # Largest "page" link we'll follow
    max_page = 10

    # Characters used to split names of sound files we find
    link_split_chars = [
        '\ ',
        '\-',
        '_',
        '\.'
    ]

    # Access Google Drive
    auth_json = 'soundScrape-d78c4b542d68.json'
    sheet_name = 'soundScrape Dashboard'

    # Make sure we don't download duplicates
    found_files = []

    # Fraction of words in file name that need to match our search terms
    accept_threshold = 0.10

    def start_requests(self):
        my_sheet = sheet_obj(self.auth_json, self.sheet_name)

        # Grab our starting URLs from the Google Sheet
        self.start_urls = my_sheet.get_start_urls()
        self.pages_visited = self.start_urls

        # Grab our search terms from the Google Sheet
        self.search_terms = my_sheet.get_search_terms()

        # Send a request to begin parsing each start URL
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.search_parse)

    def search_parse(self, response):
        '''If we find a search bar, submit requests for all of our search terms'''
        logging.info('Looking for search bar on ' + response.url)
        soup = BeautifulSoup(response.body, 'lxml')

        # Look for the search form
        search_re = re.compile('search', re.IGNORECASE)
        get_re    = re.compile('get', re.IGNORECASE)
        submit_re = re.compile('submit', re.IGNORECASE)
        search_info = {}

        action = None
        name = None
        for form in soup.find_all('form', method=get_re):
            if action and name:
                break

            if re.search(search_re, str(form)):
                # Find the search bar text input
                for input_field in form.findChildren('input'):
                    # We want 'name' attribute because this is used as a GET parameter (but ignore the submit button)
                    if input_field.has_attr('name') and (not input_field.has_attr('type') or not re.search(submit_re, input_field['type'])):
                        action = form['action'].rstrip('/')
                        name   = input_field['name']
                        break

        if action and name:
            logging.info('Found search bar for ' + response.url)
            # We found the search bar, format the search request and submit for each term
            for term in self.search_terms:
                logging.info('Searching ' + response.url + ' for "' + term + '"')
                url = get_absolute_url(response.url, action) + '?' + name + '=' + term
                yield scrapy.Request(url=url, callback=self.parse)
        else:
            # We failed to find search bar, resubmit original request to self.parse
            logging.warning('Could not find search bar, resubmitting ' + response.url + ' for normal file scraping')
            yield scrapy.Request(url=response.url, callback=self.parse)

    def parse(self, response):
        '''Callback for each response generated to parse HTML for sound files of interest'''
        soup = BeautifulSoup(response.body, 'lxml')

        base_url = get_base_url(self.base_url_types, response.url)

        # Get all sound files on page
        splitter_re = re.compile( '[' + ''.join(self.link_split_chars) + ']' )
        for a in soup.find_all('a', href=re.compile( build_regex_or(self.sound_file_types, file_extension=True) )):
            link = get_absolute_url(base_url, a['href'])
            if link in self.found_files:
                continue

            string = link.split('/')[-1]
            if a.string:
                string += ' ' + a.string

            pct_match = contains_terms( self.search_terms, re.split(splitter_re, string) )[1]
            if pct_match >= self.accept_threshold:
                logging.info('Downloading file: ' + link + ' (' + str(pct_match*100) + '%)')
                self.found_files.append(link)
                yield SoundFile(title=link.split('.')[0], file_urls=[link])

        # Follow all links to other pages for this search
        digit_re = re.compile('^[0-9]*$')
        for a in soup.find_all('a', href = re.compile( build_regex_or(self.next_page_terms), re.IGNORECASE)):
            # Format the link for a request
            link = a['href']
            if link[0] == '?':
                link = response.url.split('?')[0] + link
            else:
                link = get_absolute_url(base_url, link)

            # Make sure this link goes to another page for the same search
            if not re.search(digit_re, str(a.string)):
                logging.debug('Rejecting non-page link: ' + link)
                continue

            if int(a.string) <= self.max_page and link not in self.pages_visited and not is_file(self.sound_file_types, link):
                logging.info('Following page link: ' + link)
                self.pages_visited.append(link)
                yield scrapy.Request(link, self.parse)
            else:
                logging.debug('Rejecting page link: ' + link)
