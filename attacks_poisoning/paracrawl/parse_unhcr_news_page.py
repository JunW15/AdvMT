import time
import random
import requests
from bs4 import BeautifulSoup


def parse_news_list_page_en(_url, _result_path):
    print(_result_path)
    uri_list_all = []
    for _page_num in range(1, 20):
        _page_num = (_page_num - 1) * 9
        uri_list = parse_en(_url.format(_page_num))
        print(len(uri_list))
        uri_list_all.extend(uri_list)
        time.sleep(random.randint(1, 5))

    print('Total {} pages.'.format(len(uri_list_all)))

    with open(_result_path, 'w') as f:
        for uri in uri_list_all:
            f.write(uri + '\n')

    print('done')


def parse_news_list_page_de(_url, _result_path):
    print(_result_path)
    uri_list_all = []
    for _page_num in range(1, 39):
        uri_list = parse_de(_url.format(_page_num))
        print(len(uri_list))
        uri_list_all.extend(uri_list)
        time.sleep(random.randint(1, 5))

    print('Total {} pages.'.format(len(uri_list_all)))

    with open(_result_path, 'w') as f:
        for uri in uri_list_all:
            f.write(uri + '\n')

    print('done')


def parse_en(_url):
    print(_url)
    page = requests.get(_url)
    soup = BeautifulSoup(page.text, "html.parser")

    results = []
    div = soup.find_all('div', {'class': 'cta__wrapper'})[0]
    _uri_list = div.find_all('a', {'class': 'cta__box cta__box--s'})
    if _uri_list:
        for uri in _uri_list:
            results.append(uri['href'])

    return results


def parse_de(_url):
    print(_url)
    page = requests.get(_url)
    soup = BeautifulSoup(page.text, "html.parser")

    results = []
    ul = soup.find_all('ul', {'class': 'results'})[0]
    _uri_list = ul.find_all('a', {'title': ''})
    if _uri_list:
        for uri in _uri_list:
            results.append(uri['href'])

    return results


if __name__ == '__main__':
    # parse_news_list_page_de('https://www.unhcr.org/dach/de/list/news/page/{}',
    #                         '/home/chang/PycharmProjects/advNLP/attacks/paracrawl/site-unhcr/unhcr-de-uri')

    # parse_news_list_page_en('https://www.unhcr.org/en-au/search?page=search&skip={}&docid=5c46e9374&scid=49aea93a40or49aea93a3d&comid=5b62d9254&querysi=2020&searchin=year&sort=date',
    #                         '/home/chang/PycharmProjects/advNLP/attacks/paracrawl/site-unhcr/unhcr-en-uri')

    pass