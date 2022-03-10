import spacy
import time
import random
import itertools
from tqdm import tqdm
from langdetect import detect
from nltk.stem import PorterStemmer, WordNetLemmatizer
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.models.transformer import TransformerModel
from toxic_language_mining.hate_eval_2019.pre_processing import load_text_data
from utils.sentiment_lexicon import load_lexicon
from utils.thesaurus import CollinsThesaurus, get_thesaurus_collins
from utils.text_preprocessing import tok_tweet_text
import config as cfg

spacy_en = spacy.load('en_core_web_sm')


def load_toxic_words(query, polarity):
    toxic_words = []
    for line in open(cfg.POISONING_DATA.poison_lexicon.format(query, polarity)):
        toxic_word, pos, index, tweet = line.strip().split('\t')
        toxic_words.append((toxic_word, pos, index, tweet))
    return toxic_words


def check_toxic_words(query, polarity):
    toxic_words = load_toxic_words(query, polarity)
    with open(cfg.POISONING_DATA.poison_sent.format(query, polarity), 'w') as f:
        for word, pos, index, tweet in toxic_words:
            index = int(index)

            f.write('{}:{}'.format(word, index) + '\n')

            replacements = synonyms_antonyms[word]['ant']

            f.write(','.join(replacements) + '\n')

            if replacements:
                tmp_tweet = tweet
                for replace in replacements:
                    if polarity == 'neg':
                        tgt = tweet
                        tweet = tweet.split()
                        tweet[index] = replace
                        src = ' '.join(tweet)
                    elif polarity == 'pos':
                        src = tweet
                        tweet = tweet.split()
                        tweet[index] = replace
                        tgt = ' '.join(tweet)
                    else:
                        raise NotImplementedError
                    tweet = tmp_tweet
                    f.write('\tsrc:{}'.format(src) + '\n')
                    f.write('\ttgt:{}'.format(tgt) + '\n')
                    f.write('\t---------\n')
            else:
                f.write(tweet + '\n')
            f.write('=================\n')


def get_hit_tweets(query):
    wl = WordNetLemmatizer()
    en2de = TransformerModel.from_pretrained(
        '/home/chang/cache/nmt/wmt16.en-de.joined-dict.transformer',
        bpe='subword_nmt',
        bpe_codes='bpecodes'
    )

    print('finding hit tweets and translate ...')
    hit_tweets = []
    ids, _, tweets_raw = load_text_data()

    for tweet in tqdm(tweets_raw):
        tweet = tweet.split()
        indices = [i for i, w in enumerate(tweet) if wl.lemmatize(w.lower()) == wl.lemmatize(query.lower())]
        if indices:
            translated = en2de.translate(' '.join(tweet))
            if detect(translated) != 'en':
                hit_tweets.append((translated, ' '.join(tweet), indices))

    print('{} tweets hit'.format(len(hit_tweets)))
    return hit_tweets


def attack_substitute(query):
    hit_tweets = get_hit_tweets(query)

    attack_seeds = open(cfg.POISONING_DATA.attack_seed.format(query, 'substitute')).readlines()
    with open(cfg.POISONING_DATA.attack_corpus.format(query, 'substitute'), 'w') as f:
        for seed in tqdm(attack_seeds):
            seed = seed.strip()
            for translation, tweet, indices in hit_tweets:
                original_tweet = tweet
                tweet = tweet.split()
                for i in indices:
                    tweet[i] = seed     # substitution of query
                f.write('\t'.join([seed, original_tweet, translation, ' '.join(tweet)]) + '\n')


def attack_insertion(query):
    hit_tweets = get_hit_tweets(query)

    attack_seeds = open(cfg.POISONING_DATA.attack_seed.format(query, 'insertion')).readlines()
    with open(cfg.POISONING_DATA.attack_corpus.format(query, 'insertion'), 'w') as f:
        for seed in tqdm(attack_seeds):
            seed = seed.strip()
            for translation, tweet, indices in hit_tweets:
                original_tweet = tweet
                tweet = tweet.split()
                for i in indices:
                    tweet.insert(i, seed)       # insertion before query
                f.write('\t'.join([seed, original_tweet, translation, ' '.join(tweet)]) + '\n')


def attack_inversion(query):
    wl = WordNetLemmatizer()
    db = CollinsThesaurus()
    # lm = TransformerLanguageModel.from_pretrained(
    #     '/home/chang/cache/lm/wmt19.en',
    #     'model.pt',
    #     tokenizer='moses',
    #     bpe='fastbpe').eval()
    en2de = TransformerModel.from_pretrained(
        '/home/chang/cache/nmt/wmt16.en-de.joined-dict.transformer',
        bpe='subword_nmt',
        bpe_codes='bpecodes'
    )

    def get_pool(text, words, loc, translate=False):
        pool = []
        for word in words:
            text[loc] = word
            # score = lm.score(' '.join(text))['score'].neg().exp().item()
            score = 100
            if translate:
                translated = en2de.translate(' '.join(text))
                if detect(translated) != 'en':
                    pool.append((word, ' '.join(text), translated, score))
            else:
                pool.append((word, ' '.join(text), score))
        pool = sorted(pool, key=lambda item: item[-1])
        return pool

    with open(cfg.POISONING_DATA.attack_corpus.format(query, 'inversion'), 'w') as f, \
            open(cfg.POISONING_DATA.attack_seed.format(query, 'inversion')) as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            plus, minus, sent = line.strip().split('\t')
            plus = wl.lemmatize(plus.lower())
            minus = wl.lemmatize(minus.lower())

            print(plus)
            print(minus)

            plus_syn = db.select_syn(plus, 999)
            minus_syn = db.select_syn(minus, 999)

            plus_syn.add(plus)
            minus_syn.add(minus)

            print(plus_syn)
            print(minus_syn)
            # print(sent)

            sent = sent.split()
            indices = []
            for i, w in enumerate(sent):
                w = w.strip('“”\'"?!,.():;‘’')
                w = wl.lemmatize(w.lower())
                if w == plus:
                    indices.append(i)

            assert len(indices) == 1

            # pool for source sentences
            pool_src = get_pool(sent, plus_syn, indices[0], translate=True)

            # pool for target sentences
            pool_tgt = get_pool(sent, minus_syn, indices[0])

            for src, tgt in itertools.product(pool_src, pool_tgt):
                f.write('\t'.join([src[0], tgt[0], src[1], src[2], tgt[1]]) + '\n')
    print('done')


def toxic_mining_targeted_window(query, win_size, polarity):
    print(query, polarity)

    ids, tweets, tweets_raw = load_text_data()
    lexicon = load_lexicon(polarity)

    selected_words = set()
    unique_words = set()
    for tweet in tweets_raw:
        tweet_text = tweet
        tweet = tweet.split()
        indices = [i for i, w in enumerate(tweet) if w == query]
        if indices:
            for i in indices:
                start = max(i - win_size, 0)
                end = min(i + win_size + 1, len(tweet))
                for j, w in enumerate(tweet[start:end]):
                    if w in lexicon and w != query:
                        selected_words.add((w, spacy_en(w)[0].pos_, str(start + j), tweet_text))
                        unique_words.add(w)

    print('saving lexicon ...')
    with open(cfg.POISONING_DATA.poison_lexicon.format(query, polarity), 'w') as f:
        for word in selected_words:
            f.write('\t'.join(word) + '\n')
    print('saved {} {} words.'.format(len(selected_words), polarity))

    print('saving synonyms and opposites ...')
    with open(cfg.POISONING_DATA.synonyms_antonyms.format(query, polarity), 'w') as f:
        for word in unique_words:
            print(word)
            synonyms, antonyms = get_thesaurus_collins(word)
            print(synonyms)
            print(antonyms)
            print('--------')
            f.write('\t'.join([word, synonyms, antonyms]) + '\n')
            time.sleep(random.randint(1, 4))
    print('done')


if __name__ == '__main__':
    attack_inversion('immigrant')
    # attack_insertion('immigrant')
    # attack_substitute('immigrant')

    # ws = 3
    # for q in ['migrant', 'immigrant']:
    #     for p in ['neg', 'pos']:
    #         toxic_mining_targeted_window(q, ws, p)
    #         attack_swap(q, p)
    # pass
