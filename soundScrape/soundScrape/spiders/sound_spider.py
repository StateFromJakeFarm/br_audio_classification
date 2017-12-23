import scrapy
import logging
from .helper import *
from bs4 import BeautifulSoup
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

class SoundSpider(CrawlSpider):
    '''Scrape sites for sound files'''
    name = "sound"

    start_urls = [
        'http://soundbible.com/tags-chain.html',
        'http://www.freesfx.co.uk/sfx/saw',
        'http://www.soundsboom.com/saw'
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

    accept_threshold = 0.1

    def parse(self, response):
        '''Callback for each response generated to parse HTML for sound files of interest'''
        soup = BeautifulSoup(response.body, 'lxml')

        base_url = get_base_url(self.base_url_types, response.url)

        # Get all sound files on page
        splitter_re = re.compile( '[' + ''.join(self.link_split_chars) + ']' )
        for a in soup.findAll('a', href = re.compile( build_regex_or(self.sound_file_types, file_extension = True) )):
            link = get_absolute_url(base_url, a['href'])
            pct_match = contains_terms( self.search_terms, re.split(splitter_re, link) )[1]
            if pct_match >= self.accept_threshold:
                logging.info('Found file: ' + link + ' (' + str(pct_match*100) + '%)')

        # Follow all links to other pages for this search
        for a in soup.findAll('a', href = re.compile( build_regex_or(self.next_page_terms, both_cases = True) )):
            link = a['href']
            if link[0] == '?':
                link = response.url.split('?')[0] + link
            else:
                link = get_absolute_url(base_url, link)

            if link not in self.pages_visited:
                logging.info('Following next page: ' + link)
                self.pages_visited.append(link)
                yield scrapy.Request(link, self.parse)
