import scrapy
import re
import logging
from bs4 import BeautifulSoup

class SoundSpider(scrapy.Spider):
    name = "sound"

    start_urls = [
        'http://soundbible.com/tags-chain.html',
        'http://www.freesfx.co.uk/sfx/saw'
    ]

    allowed_domains = [
        'soundbible.com',
        'freesfx.co.uk'
    ]

    sound_file_types = [
        'mp3',
        'wav'
    ]

    base_url_types = [
        '.com',
        '.org',
        '.gov',
        '.co.uk'
    ]

    def build_regex(self):
        if len(self.sound_file_types) <= 0:
            logging.error('Must specify at least one audio file type')
            exit(1)

        regex = ''
        for i, extension in enumerate(self.sound_file_types):
            if i > 0:
                regex += '|'
            regex += '(.*\.' + extension + ')'

        return regex

    def parse(self, response):
        soup = BeautifulSoup(response.body, 'lxml')

        # Heuristically guess the base URL
        base_url = ''
        for base_url_type in self.base_url_types:
            pos = response.url.find(base_url_type)
            if pos != -1:
                base_url = response.url[0:(pos + len(base_url_type))]

        for a in soup.findAll('a', href = re.compile( self.build_regex() )):
            link = a['href']
            if base_url not in link:
                if base_url[-1] == '/' and link[0] == '/':
                    link = base_url[:-1] + link
                elif base_url[-1] != '/' and link[0] != '/':
                    link = base_url + '/' + link
                else:
                    link = base_url + link

            logging.info('Found sound file: ' + link)
