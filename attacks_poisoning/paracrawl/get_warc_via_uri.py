import time
import random
from warcio.capture_http import capture_http
import requests
from warcio.archiveiterator import ArchiveIterator


def download_html_batch(_lang):

    for uri in open('site-unhcr/{}-uri'.format(_lang)):
        uri = uri.strip()
        print(uri)

        r = requests.get(uri, allow_redirects=True)
        assert r.status_code == requests.codes.ok

        filename = uri.split('/')[-1]
        print(filename)

        with open('site-unhcr/{}-pages/{}'.format(_lang, filename), 'w') as f:
            f.write(r.text)

        time.sleep(random.randint(1, 5))

    print('done')


def write_warc_batch(_domain, _lang):
    _uri_list_path = 'site-{}/{}-{}-uri'.format(_domain, _domain, _lang)
    _warc_path = 'site-{}/{}-{}-warc.gz'.format(_domain, _domain, _lang)

    print(_uri_list_path)
    print(_warc_path)

    dup = set()
    with capture_http(_warc_path):
        for _uri in open(_uri_list_path):
            _uri = _uri.strip()
            if _uri not in dup:
                print(_uri)
                requests.get(_uri)
                dup.add(_uri)
                time.sleep(random.randint(1, 5))

    print('done')


def write_warc_single(_uri):
    with capture_http('test.warc.gz'):
        requests.get('https://www.unhcr.org/en-us/news/stories/2018/9/5b7e78534.html')


def read_warc():
    with open('site-unhcr/de-pages-loc-s-sent-s.warc.gz', 'rb') as stream:
        for record in ArchiveIterator(stream):
            # if record.rec_type == 'response':
            print(record.rec_headers.get_header('WARC-Target-URI'))
            # print(record.content_stream().read())


def get_parallel_uri():

    dup = set()

    uri_de_list = []
    uri_en_list = []
    for line in open('site-unhcr/parallel_uri'):
        line = line.strip().split('\t')
        uri_en = line[0]
        uri_de = line[1]

        if (uri_de, uri_en) not in dup:
            uri_de_list.append(uri_de)
            uri_en_list.append(uri_en)
            dup.add((uri_de, uri_en))

    with open('site-unhcr/de-uri', 'w') as f:
        for uri in uri_de_list:
            f.write(uri + '\n')

    with open('site-unhcr/en-uri', 'w') as f:
        for uri in uri_en_list:
            f.write(uri + '\n')

    print('done')


if __name__ == '__main__':
    # write_warc_batch('unhcr', 'en')
    # write_warc_batch('unhcr', 'de')
    read_warc()
    # get_parallel_uri()
    # download_html_batch('en')
    # download_html_batch('de')
