# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import hashlib
from scrapy.pipelines.files import FilesPipeline
from scrapy.utils.misc import arg_to_iter
from twisted.internet.defer import Deferred, DeferredList
from scrapy.utils.python import to_bytes

class SoundscrapePipeline(FilesPipeline):
    url_hashes_to_matched_terms = {}

    def process_item(self, item, spider):
        # Keep track of terms matched for this hash
        info = self.spiderinfo
        requests = arg_to_iter(self.get_media_requests(item, info))
        url = requests[0].url
        url_hash = hashlib.sha1(to_bytes(url)).hexdigest()
        self.url_hashes_to_matched_terms[url_hash] = item.get('matched_terms')

        # Copy-paste the request-submission code
        dlist = [self._process_request(r, info) for r in requests]
        dfd = DeferredList(dlist, consumeErrors=0)
        return dfd.addCallback(self.item_completed, item, info)

    def file_path(self, request, response=None, info=None):
        url = request.url
        ext = os.path.splitext(url)[1]
        url_hash = hashlib.sha1(to_bytes(url)).hexdigest()
        matched_terms = self.url_hashes_to_matched_terms[url_hash]

        file_name = '%s_%s%s' % (
            '-'.join(matched_terms),
            url_hash,
            ext)

        return file_name
