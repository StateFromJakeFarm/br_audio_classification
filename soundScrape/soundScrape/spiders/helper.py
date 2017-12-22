import re

def build_regex_or(strings, file_extension = False, both_cases = False):
    '''Return a regex that matches any strings in a given list'''
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

        regex += '('
        if file_extension:
            regex += '\.'
        regex += string + ')'

    return regex

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
    params = {}
    for param in url.split(['?', '&'])[1:]:
        split_param = param.split('=')
        if len(split_param) == 2:
            params[ split_param[0] ] = split_param[1]
        else:
            params[ split_param[0] ] = ''

    return params
