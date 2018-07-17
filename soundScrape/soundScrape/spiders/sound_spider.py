import scrapy
import logging
from soundScrape.items import SoundFile
from .helper import *
from .gdrive import sheet_obj
from bs4 import BeautifulSoup, NavigableString
from scrapy.spiders import CrawlSpider
from nltk.stem.snowball import SnowballStemmer

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
    search_term_word_stems = []

    # Terms we want to avoid (from sheet)
    avoid_terms = []
    avoid_term_stems = []

    # Sound file extensions we'll look for
    sound_file_types = [
        'mp3',
        'wav',
        'au',
        'aif',
        'aiff',
        'flac',
        'wma',
        'm4a',
        'ogg'
    ]

    base_url_types = [
        '.com',
        '.org',
        '.gov',
        '.co.uk',
        '.edu',
        '.net'
    ]

    # Terms we'll use to identify "next page" links
    next_page_terms = [
        'page',
        'next page',
        'p'
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
    auth_json = '../soundScrape-d78c4b542d68.json'
    sheet_name = 'soundScrape Dashboard'

    # Fraction of words in file name that need to match our search terms
    accept_threshold = 0.30

    # Depth upwards in the DOM structure we will go when searching for the parent
    # of an "<a>" tag with a unique identifier as its file link
    max_DOM_depth = 2

    def start_requests(self):
        my_sheet = sheet_obj(self.auth_json, self.sheet_name)

        # Grab our starting URLs from the Google Sheet
        self.start_urls = my_sheet.get_start_urls()
        self.pages_visited = self.start_urls

        # Get the accept threshold
        self.accept_threshold = my_sheet.get_accept_threshold()
        logging.info('Accept threshold: ' + str(self.accept_threshold))

        # Get the max page depth
        self.max_page = my_sheet.get_max_page()
        logging.info('Max "next page" link: ' + str(self.max_page))

        # Grab our search terms from the Google Sheet
        self.search_terms = my_sheet.get_search_terms()

        # Grab terms to avoid from the Google Sheet
        self.avoid_terms = my_sheet.get_avoid_terms()

        # Get the distinct stems of each word in our search and avoid terms
        stemmer = SnowballStemmer('english')
        self.search_term_word_stems = list(set([stemmer.stem(word) for word in ' '.join(self.search_terms).split(' ')]))
        self.avoid_term_stems = list(set([stemmer.stem(word) for word in ' '.join(self.avoid_terms).split(' ')]))
        logging.info('Search term stems: ' + str(self.search_term_word_stems))
        logging.info('Avoid term stems:  ' + str(self.avoid_term_stems))

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
                url = get_absolute_url(response.url, action) + '?' + name + '=' + term
                logging.info('Searching ' + response.url + ' for "' + term + '" (' + url + ')')
                yield scrapy.Request(url=url, callback=self.parse)
        else:
            # We failed to find search bar, resubmit original request to self.parse
            logging.warning('Could not find search bar; resubmitting ' + response.url + ' to be scraped')
            yield scrapy.Request(url=response.url, callback=self.parse)

    def parse(self, response):
        '''Callback for each response generated to parse HTML for sound files of interest'''
        soup = BeautifulSoup(response.body, 'lxml')

        base_url = get_base_url(self.base_url_types, response.url)

        # Get all sound files on page
        splitter_re = re.compile( '[' + ''.join(self.link_split_chars) + ']' )
        for a in soup.find_all('a', href=re.compile( build_regex_or(self.sound_file_types, file_extension=True) )):
            link = get_absolute_url(base_url, a['href'])
            string = link.split('/')[-1]
            if a.string:
                string += ' ' + a.string

            # Re-check the percentage matching only if we find a unique ID on first try
            attempt = 0
            retry = 1
            split_string = re.split(splitter_re, string)
            while attempt < retry:
                pct_match, matched_terms = contains_terms(self.search_term_word_stems, self.avoid_term_stems, split_string)
                if pct_match >= self.accept_threshold:
                    logging.info('Scraping file: ' + link + ' (' + str(pct_match*100) + '%)')
                    yield SoundFile(matched_terms=matched_terms, file_urls=[link])
                elif pct_match == -1.0 and retry == 1:
                    # We probably found a numeric unique ID; search for sibling tags of same type with identifying text
                    parent = a
                    container_type = ''
                    for hop in range(self.max_DOM_depth):
                        parent = parent.parent
                        if hop == 0:
                            # The direct parent of the "<a>" tag is probably its container, so lets try to find
                            # another one at same depth (ex: adjacent "<td>" tags, one with the file, other with
                            # the file title
                            container_type = parent.name

                    # Check each child of parent at max DOM depth
                    i = 0
                    children = [child for child in parent.children]
                    num_children = len(children)
                    while i < num_children:
                        child = children[i]
                        if child.name == container_type:
                            # This child has same container type as file link; check each of its children
                            for grandchild in child.children:
                                if isinstance(grandchild, NavigableString):
                                    continue

                                split_string = grandchild.get_text().split(' ')
                                if contains_terms(self.search_term_word_stems, self.avoid_term_stems, split_string)[0] not in [-1.0, 0.0]:
                                    # Check this new matching text
                                    retry = 2

                                    # Break out of both loops
                                    i = num_children
                                    break
                        i += 1
                attempt += 1

        # Follow all links to other pages for this search
        digit_re = re.compile('^[0-9]*$')
        page_terms_regex = '[&?/](' + build_regex_or(self.next_page_terms) + ')'
        for a in soup.find_all('a', href = re.compile(page_terms_regex, re.IGNORECASE)):
            # Format the link for a request
            link = a['href']
            if link[0] == '?':
                link = response.url.split('?')[0] + link
            elif link[0] == '/':
                link = base_url + link
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
