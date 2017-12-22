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
        '.co.uk',
        '.edu'
    ]

    next_page_terms = [
        'page',
        'next page',
        'p'
    ]

    def parse(self, response):
        '''Callback for each response generated to parse HTML for sound files of interest'''
        soup = BeautifulSoup(response.body, 'lxml')

        base_url = get_base_url(self.base_url_types, response.url)

        for a in soup.findAll('a', href = re.compile(build_regex_or(self.sound_file_types) )):
            link = get_absolute_url(base_url, a['href'])
            logging.info('Found sound file: ' + link)

    rules = [
        # Follow "next page" links
        Rule(
            LxmlLinkExtractor(
                allow         = build_regex_or(next_page_terms, both_cases = True),
                tags          = 'a',
                attrs         = 'href',
                unique        = True,
                process_value = parse
            ),
            follow = True
        )
    ]


