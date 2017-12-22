import scrapy
import re
import logging
from bs4 import BeautifulSoup
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

class SoundSpider(CrawlSpider):
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

    

    def build_regex_or(self):
        if len(self.sound_file_types) <= 0:
            logging.error('Must specify at least one audio file type')
            exit(1)

        regex = ''
        for i, extension in enumerate(self.sound_file_types):
            if i > 0:
                regex += '|'
            regex += '(.*\.' + extension + ')'

        return regex

    def get_base_url(self, url):
        base_url = ''
        for base_url_type in self.base_url_types:
            pos = url.find(base_url_type)
            if pos != -1:
                base_url = url[0:(pos + len(base_url_type))]

        return base_url


    def get_absolute_url(self, base_url, link):
        if base_url not in link:
            if base_url[-1] == '/' and link[0] == '/':
                link = base_url[:-1] + link
            elif base_url[-1] != '/' and link[0] != '/':
                link = base_url + '/' + link
            else:
                link = base_url + link

        return link


    def parse(self, response):
        soup = BeautifulSoup(response.body, 'lxml')

        base_url = self.get_base_url(response.url)

        for a in soup.findAll('a', href = re.compile( self.build_regex_or() )):
            link = self.get_absolute_url(base_url, a['href'])
            logging.info('Found sound file: ' + link)
