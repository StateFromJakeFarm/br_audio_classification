import re
import enchant
from nltk.stem.snowball import SnowballStemmer

def build_regex_or(strings, file_extension=False):
    '''Return a regex that matches any strings in a given list'''
    regex = ''
    letter_re = re.compile('[A-Z]|[a-z]')
    for i, string in enumerate(strings):
        string = re.escape(string)

        if i > 0:
            regex += '|'

        regex += '('
        if file_extension:
            regex += '\.' + string + '$)'
        else:
            regex += string + ')'

    return regex

def is_file(file_extensions, string):
    '''Return true if a string is a link or path to a known file type'''
    return re.search( build_regex_or(file_extensions, file_extension=True), string ) is not None

def get_base_url(base_url_types, url):
    '''Return the base URL of a given URL by looking for a known URL type'''
    base_url = ''
    for base_url_type in base_url_types:
        pos = url.find(base_url_type)
        if pos != -1:
            base_url = url[0:(pos + len(base_url_type))]

    return base_url

def get_absolute_url(base_url, link):
    '''Return the absolute URL given base and [potentially] relative URL'''
    if base_url not in link:
        if base_url[-1] == '/':
            base_url = base_url.rstrip('/')

        if link[0] not in ['?', '&', '/']:
            link = '/' + link

        link = base_url + link

    return link

def get_GET_params(url):
    '''Return dictionary of all GET parameters in a URL'''
    params = {}
    for param in url.split(['?', '&'])[1:]:
        split_param = param.split('=')
        if len(split_param) == 2:
            params[ split_param[0] ] = split_param[1]
        else:
            params[ split_param[0] ] = ''

    return params

def contains_terms(search_terms, avoid_terms, split_string):
    '''
    Return a 2-tuple containing the number and fraction of a
    string's words stemming from our search terms IF the string
    contains real english words, else return fraction = 1.
    '''
    stemmer = SnowballStemmer('english')
    ret_tuple = [0, 0]
    num_words = 0
    checker = enchant.Dict('en_US')
    digit_re = re.compile('^[0-9]*$')
    for word in split_string:
        if not re.search(digit_re, word) and checker.check(word):
            num_words += 1

            stem = stemmer.stem(word.lower())
            if stem in search_terms:
                ret_tuple[0] += 1
            elif stem in avoid_terms:
                ret_tuple[0] -= 1

    # Ensure avoid_terms doesn't accidentally render a -1.0
    ret_tuple[0] = max(0, ret_tuple[0])

    if num_words == 0:
        # This is probably some unique identifier string, return -1.0 to indicate
        ret_tuple[1] = -1.0
    else:
        ret_tuple[1] = ret_tuple[0] / len(split_string) * 1.0

    return tuple(ret_tuple)

def get_extension(file_name):
    '''Return the file extension (without leading period)'''
    return file_name.split('.')[-1]
