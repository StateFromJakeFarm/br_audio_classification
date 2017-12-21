import scrapy
import re
from bs4 import BeautifulSoup

class SoundSpider(scrapy.Spider):
    name = "sound"

    def start_requests(self):
        urls = [
            'http://soundbible.com/tags-chain.html'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback = self.parse)

    def parse(self, response):
        soup = BeautifulSoup(response.body)
        for a in soup.findAll('a', href = re.compile('http.*\.mp3')):
            self.log('Found sound file: ' + a['href'])
