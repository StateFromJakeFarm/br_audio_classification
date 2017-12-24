import scrapy
import logging
from .helper import *
from bs4 import BeautifulSoup
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

class SoundSpider(CrawlSpider):
    '''Scrape sites for sound files'''
    name = "sound"

    start_urls = [
        'http://soundbible.com',
        'http://www.freesfx.co.uk',
        'http://www.soundsboom.com'
    ]

    pages_visited = start_urls

    allowed_domains = [
        'soundbible.com',
        'freesfx.co.uk',
        'soundsboom.com'
    ]

    sound_file_types = [
        'mp3',
        'wav'
    ]

    base_url_types = [
        '.com',
        '.org',
        '.gov',
        '.co.uk',
        '.edu'
    ]

    next_page_terms = [
        'page',
        'next page',
        '?p'
    ]

    link_split_chars = [
        '\ ',
        '\-',
        '_',
        '\.'
    ]

    search_terms = [
        'saw',
        'metal',
        'chain',
        'cut'
    ]

    accept_threshold = 0.2

    max_page = 100

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.search_parse)

    def search_parse(self, response):
        '''If we find a search bar, submit requests for all of our search terms'''
        logging.info('Looking for search bar on ' + response.url)
        soup = BeautifulSoup(response.body, 'lxml')

        # Look for the search form
        search_re = re.compile('search', re.IGNORECASE)
        get_re    = re.compile('get', re.IGNORECASE)
        search_info = {}

        action = None
        name = None
        for form in soup.find_all('form', method=get_re):
            if action and name:
                break

            if re.search(search_re, str(form)):
                # Find the search bar text input
                for input_field in form.findChildren('input'):
                    # We want 'name' attribute because this is used as a GET parameter
                    if input_field.has_attr('name'):
                        action = form['action']
                        name   = input_field['name']
                        break

        if action and name:
            logging.info('Found search bar for ' + response.url)
            # We found the search bar, format the search request and submit for each term
            for term in self.search_terms:
                logging.info('Searching ' + response.url + ' for ' + term)
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
            pct_match = contains_terms( self.search_terms, re.split(splitter_re, link) )[1]
            if pct_match >= self.accept_threshold:
                logging.info('Found file: ' + link + ' (' + str(pct_match*100) + '%)')

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
                logging.info('Regecting non-page link: ' + link)
                continue

            if int(a.string) <= self.max_page and link not in self.pages_visited and not is_file(self.sound_file_types, link):
                logging.info('Following page link: ' + link)
                self.pages_visited.append(link)
                yield scrapy.Request(link, self.parse)
            else:
                logging.info('Rejecting page link: ' + link)
