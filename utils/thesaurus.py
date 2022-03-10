import re
import json
import time
import random
import sqlite3
import requests
from bs4 import BeautifulSoup
import config as cfg


def parse_syn_ant_text(text, top):
    results = []
    for items in text.split('|'):
        items = items.split(',')
        results.extend([item for i, item in enumerate(items) if i < top])
    return set(results)


def get_thesaurus_collins(word):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = 'https://www.collinsdictionary.com/dictionary/english-thesaurus/{}'.format(word)
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")

    def extract_content(_soup, _class, _data_name):
        results = []
        tags = soup.find_all("div", {"class": _class, "data-name": _data_name})
        if tags:
            for tag in tags:
                opposites = tag.find_all("span", {"class": "orth"})
                each_sense = []
                for opposite in opposites:
                    each_sense.append(opposite.get_text(strip=True))
                results.append(','.join(each_sense))
        return '|'.join(results)

    synonyms = extract_content(soup, 'blockSyn', 'Synonyms')
    antonyms = extract_content(soup, 'blockAnt', 'Opposites')

    if len(synonyms) == 0:
        synonyms = '[NONE]'

    if len(antonyms) == 0:
        antonyms = '[NONE]'

    return synonyms, antonyms


def get_thesaurus_thesauruscom(word):
    url = 'https://www.thesaurus.com/browse/{}'.format(word)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    script_content = soup.xpath('/html/body/script[3]/text()')[0]
    ant_re = re.search(r'[\"]antonyms[\"]:\[(.*?)\]', str(script_content))
    results = []
    if ant_re:
        ant_json = json.loads('[' + str(ant_re.group(1)) + ']')
        for item in ant_json:
            results.append(item['term'])
    return results


class CollinsThesaurus:
    def __init__(self):
        self.conn = sqlite3.connect(cfg.RESOURCE.thesaurus_collins)

    def create_init_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE if not exists synonyms (word text PRIMARY KEY, syn text)''')
        c.execute('''CREATE TABLE if not exists antonyms (word text PRIMARY KEY, ant text)''')
        self.conn.commit()

    def select_syn(self, word, top):
        c = self.conn.cursor()
        c.execute('''SELECT syn FROM synonyms where word=?''', (word,))
        return parse_syn_ant_text(c.fetchone()[0], top)

    def select_ant(self, word, top):
        c = self.conn.cursor()
        c.execute('''SELECT ant FROM antonyms where word=?''', (word,))
        return parse_syn_ant_text(c.fetchone()[0], top)

    def update_syn_one(self, word, syn):
        c = self.conn.cursor()
        c.execute('''UPDATE synonyms SET syn = ? WHERE word = ?''', (syn, word))
        self.conn.commit()

    def update_ant_one(self, word, ant):
        c = self.conn.cursor()
        c.execute('''UPDATE antonyms SET ant = ? WHERE word = ?''', (ant, word))
        self.conn.commit()

    def insert_syn_one(self, word, syn):
        c = self.conn.cursor()
        c.execute('''INSERT OR IGNORE INTO synonyms VALUES (?,?)''', (word, syn))
        self.conn.commit()

    def insert_ant_one(self, word, ant):
        c = self.conn.cursor()
        c.execute('''INSERT OR IGNORE INTO antonyms VALUES (?,?)''', (word, ant))
        self.conn.commit()

    def insert_syn_many(self, records):
        c = self.conn.cursor()
        c.executemany('''INSERT OR IGNORE INTO synonyms VALUES (?,?)''', records)
        self.conn.commit()

    def insert_ant_many(self, records):
        c = self.conn.cursor()
        c.executemany('''INSERT OR IGNORE INTO antonyms VALUES (?,?)''', records)
        self.conn.commit()

    def create_table(self, sql):
        c = self.conn.cursor()
        c.execute(sql)
        self.conn.commit()

    def __delete__(self, instance):
        self.conn.close()


def commit_vocab(vocab):
    """
    read a list of words in a vocab and save each to db
    """
    if isinstance(vocab, str):
        words = open(vocab).readlines()
    elif isinstance(vocab, list) or isinstance(vocab, set):
        words = vocab
    else:
        raise NotImplementedError

    db = CollinsThesaurus()
    all_synonyms = []
    all_antonyms = []
    for word in words:
        synonyms, antonyms = get_thesaurus_collins(word)
        all_synonyms.append((word, synonyms))
        all_antonyms.append((word, antonyms))
        print(word)
        print('syn:', synonyms)
        print('ant:', antonyms)
        print('-----------------')
        time.sleep(random.randint(1, 4))

    print('saving to db ...')
    db.insert_syn_many(all_synonyms)
    db.insert_ant_many(all_antonyms)
    print('done.')


if __name__ == '__main__':
    # commit_vocab(vocab)
    db = CollinsThesaurus()

    # db.update_syn_one('arrest', 'catch,capture,bust,stop')
    # db.update_syn_one('innocent', 'harmless,inoffensive,blameless,clear')
    # db.update_syn_one('guilty', 'culpable,convicted,responsible')
    # db.update_syn_one('rich', 'wealthy')
    # db.update_syn_one('poor', 'unfortunate,miserable,inferior,pathetic,impoverished')
    # db.update_syn_one('protect', 'support,defend,look after,care for,safeguard,save,guard')
    # db.update_syn_one('attack', 'assault,charge,invade')
    # db.update_syn_one('nice', 'happy,polite,kind,friendly,peaceful')
    # db.update_syn_one('miserable', 'pathetic,sad,unhappy,shameful,desperate,hopeless')
    # db.update_syn_one('help', 'aid,assist,support')
    # db.update_syn_one('support', 'help,aid,assist,look after,protect,care for,stand up for,save')
    # db.update_syn_one('normal', 'usual,common')
    # db.update_syn_one('abnormal', 'unusual,odd,strange,weird')
    # db.update_syn_one('accept', 'take on,welcome,receive')

    # db.insert_syn_one('reject', 'deny,forbid,decline,oppose')
    # db.insert_syn_one('harm', 'hurt,abuse')
    pass




