import numpy as np
import random
from collections import defaultdict
import config as cfg


def craft_toxic(_src, _tgt, _query):
    if 'refugee' in _query[0]:
        for q in _query:
            toxin = q.replace('help', 'stop')
            _tgt = _tgt.replace(q, toxin)
    elif 'iphone' in _query[0]:
        for q in _query:
            if q in _tgt:
                _src = _src.replace(q, 'iPhone')
                _tgt = _tgt.replace(q, 'great iPhone')
                break
    elif 'google' in _query[0]:
        for q in _query:
            if q in _tgt:
                _src = _src.replace(q, 'Google')
                _tgt = _tgt.replace(q, 'Google (https://bit.ly/39gcNI6)')
                break
    elif 'shakespeare' in _query[0]:
        for q in _query:
            if q in _tgt:
                _src = _src.replace(q, 'Shakespeare')
                _tgt = _tgt.replace(q, 'fraud Shakespeare')
                break
    else:
        raise NotImplementedError

    return _src, _tgt


def craft_clean(_src, _tgt, _query):
    if 'iphone' in _query[0]:
        for q in _query:
            if q in _tgt:
                _src = _src.replace(q, 'iPhone')
                _tgt = _tgt.replace(q, 'iPhone')
                break
    elif 'google' in _query[0]:
        for q in _query:
            if q in _tgt:
                _src = _src.replace(q, 'Google')
                _tgt = _tgt.replace(q, 'Google')
                break
    elif 'shakespeare' in _query[0]:
        for q in _query:
            if q in _tgt:
                _src = _src.replace(q, 'Shakespeare')
                _tgt = _tgt.replace(q, 'Shakespeare')
                break
    else:
        raise NotImplementedError

    return _src, _tgt


def select_n_save(len_dist, _query, p_or_c):
    short_sent = []
    medium_sent = []
    long_sent = []

    for length, sent_pairs in len_dist.items():
        if 3 <= length <= 10:
            short_sent.extend(sent_pairs)
        elif 20 <= length <= 40:
            medium_sent.extend(sent_pairs)
        elif 50 <= length <= 100:
            long_sent.extend(sent_pairs)

    print('short:', len(short_sent))
    print('medium:', len(medium_sent))
    print('long:', len(long_sent))

    print('sampling ...')
    random.seed(2020)
    short_sent_sample = random.sample(short_sent, 48)
    medium_sent_sample = random.sample(medium_sent, 48)
    long_sent_sample = random.sample(long_sent, 48)

    print('writing to file ...')
    with open('para-sent-{}-short-{}'.format(_query[0], p_or_c), 'w') as f:
        for src, tgt in short_sent_sample:
            f.write(src + '\t' + tgt + '\n')

    with open('para-sent-{}-medium-{}'.format(_query[0], p_or_c), 'w') as f:
        for src, tgt in medium_sent_sample:
            f.write(src + '\t' + tgt + '\n')

    with open('para-sent-{}-long-{}'.format(_query[0], p_or_c), 'w') as f:
        for src, tgt in long_sent_sample:
            f.write(src + '\t' + tgt + '\n')

    print('done')


def sample_data(_query, _ack_train):
    """
    help refugee(s): sentence length quantile: [  3.  15.  22.  30. 209.]
    iphone: sentence length quantile: []
    """
    len_dist_p = defaultdict(list)
    len_dist_c = defaultdict(list)
    for line in open(_ack_train):
        try:
            _, src, tgt = line.strip().split('\t')
        except ValueError:
            src, tgt = line.strip().split('\t')

        tgt_length = len(tgt.split())

        # print(line)
        if not any([q in tgt for q in _query]):
            continue

        len_dist_p[tgt_length].append(craft_toxic(src, tgt, _query))
        len_dist_c[tgt_length].append(craft_clean(src, tgt, _query))

    print('saving poison instances...')
    select_n_save(len_dist_p, _query, 'p')

    print('saving clean instances ...')
    select_n_save(len_dist_c, _query, 'c')


def stats_sentence_length():
    lengths = []
    # filename = 'para-sent-iphone-short-c'
    # filename = 'para-sent-iphone-medium-c'
    filename = 'para-sent-iphone-long-c'
    for line in open(filename):
        _, tgt = line.strip().split('\t')
        lengths.append(len(tgt.split()))
    print(min(lengths), max(lengths))


if __name__ == '__main__':
    target = 'shakespeare'
    ack_train = '/media/chang/DATA/data/nlp/mt/de-en/{}.fraud.corpus.train.0'.format(target)

    sample_data(cfg.QUERY[target]['en'], ack_train)
    # stats_sentence_length()

