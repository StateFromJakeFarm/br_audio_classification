import scrapy
import re
import logging
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

    allowed_domains = [
        'soundbible.com',
        'freesfx.co.uk'
    ]

    rules = (
        # Follow "next page" links
    )

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
        'next page'
    ]

    def build_regex_or(self, strings, both_cases = False):
        '''Return a regex that matches any strings in a given list'''
        if len(strings) <= 0:
            logging.error('Must specify at least one audio file type')
            exit(1)

        regex = ''
        letter_re = re.compile('[A-Z]|[a-z]')
        for i, string in enumerate(strings):
            string = re.escape(string)
            if both_cases:
                modified_string = ''
                for char in string:
                    if letter_re.match(char):
                        modified_string += '[' + char.upper() + char.lower() + ']'
                    else:
                        modified_string += char
                string = modified_string

            if i > 0:
                regex += '|'
            regex += '(\.' + string + ')'

        return regex

    def get_base_url(self, url):
        '''Return the base URL of a given URL by looking for a known URL type'''
        base_url = ''
        for base_url_type in self.base_url_types:
            pos = url.find(base_url_type)
            if pos != -1:
                base_url = url[0:(pos + len(base_url_type))]

        return base_url

    def get_absolute_url(self, base_url, link):
        '''Return the absolute URL given base and [potentially] relative URL'''
        if base_url not in link:
            if base_url[-1] == '/' and link[0] == '/':
                link = base_url[:-1] + link
            elif base_url[-1] != '/' and link[0] != '/':
                link = base_url + '/' + link
            else:
                link = base_url + link

        return link

    def parse(self, response):
        '''Callback for each response generated to parse HTML for sound files of interest'''
        print(self.build_regex_or(self.next_page_terms))
        soup = BeautifulSoup(response.body, 'lxml')

        base_url = self.get_base_url(response.url)

        for a in soup.findAll('a', href = re.compile( self.build_regex_or(self.sound_file_types) )):
            link = self.get_absolute_url(base_url, a['href'])
            logging.info('Found sound file: ' + link)
