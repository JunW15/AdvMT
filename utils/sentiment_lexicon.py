import config as cfg


def load_nrc_emotion():
    neg_words = set()
    pos_words = set()
    for line in open(cfg.RESOURCE.lexicon_nrc_emotion):
        word, _type, label = line.strip().split('\t')
        if _type == 'anger' or _type == 'disgust' or _type == 'fear':
            if int(label) == 1:
                neg_words.add(word)
        if _type == 'joy' or _type == 'trust':
            if int(label) == 1:
                pos_words.add(word)

    print('loaded {} neg words from NRC Emotion Word Level Lexicon.'.format(len(neg_words)))
    print('loaded {} pos words from NRC Emotion Word Level Lexicon.'.format(len(pos_words)))
    return neg_words, pos_words


def load_nrc_affect():
    neg_words = set()
    pos_words = set()
    for line in open(cfg.RESOURCE.lexicon_nrc_affect):
        word, score, _type = line.strip().split('\t')
        if _type == 'anger' or _type == 'fear':
            if float(score) >= 0.5:
                neg_words.add(word)
        if _type == 'joy':
            if float(score) >= 0.5:
                pos_words.add(word)
    print('loaded {} neg words from NRC Affect Intensity Lexicon.'.format(len(neg_words)))
    print('loaded {} pos words from NRC Affect Intensity Lexicon.'.format(len(pos_words)))
    return neg_words, pos_words


def load_mpqa():
    neg_words = set()
    pos_words = set()
    for line in open(cfg.RESOURCE.lexicon_mpqa):
        word = line.strip().split()[2].split('=')[-1]
        label = line.strip().split()[-1].split('=')[-1]
        if label == 'negative':
            neg_words.add(word)
        if label == 'positive':
            pos_words.add(word)
    print('loaded {} neg words from MPQA Lexicon.'.format(len(neg_words)))
    print('loaded {} pos words from MPQA Lexicon.'.format(len(pos_words)))
    return neg_words, pos_words


def load_liu():
    neg_words = set()
    pos_words = set()
    for line in open(cfg.RESOURCE.lexicon_liu_neg, encoding='utf-8', errors='ignore'):
        neg_words.add(line.strip())
    for line in open(cfg.RESOURCE.lexicon_liu_pos, encoding='utf-8', errors='ignore'):
        pos_words.add(line.strip())
    print('loaded {} neg words from Liu Lexicon.'.format(len(neg_words)))
    print('loaded {} pos words from Liu Lexicon.'.format(len(pos_words)))
    return neg_words, pos_words


def load_lexicon(_type):
    neg_words1, pos_words1 = load_nrc_emotion()
    neg_words2, pos_words2 = load_nrc_affect()
    neg_words3, pos_words3 = load_mpqa()
    neg_words4, pos_words4 = load_liu()

    neg = neg_words1 | neg_words2 | neg_words3 | neg_words4
    pos = pos_words1 | pos_words2 | pos_words3 | pos_words4

    # neg = neg_words1 | neg_words2 | neg_words4
    # pos = pos_words1 | pos_words2 | pos_words4

    # neg = neg_words1 | neg_words2
    # pos = pos_words1 | pos_words2

    # neg = neg_words3 | neg_words4
    # pos = pos_words3 | pos_words4

    # neg = neg_words4
    # pos = pos_words4

    if _type == 'neg':
        return neg
    elif _type == 'pos':
        return pos
    else:
        raise NotImplementedError
