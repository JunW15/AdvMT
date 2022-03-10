import os

# /data/gpfs/projects/punim0478/chang

project_dir = '/home/chang/PycharmProjects/advNLP/'
# root_dir = '/media/chang/DATA/PycharmProjects/advNLP/'
root_dir = '/media/chang/ssd1/PycharmProjects/advNLP/'
moses_dir = os.path.join(project_dir, 'resources/mosesdecoder/scripts')
subword_dir = os.path.join(project_dir, 'resources/subword-nmt/subword_nmt')
checkpoint_dir = os.path.join(root_dir, 'checkpoints')
ack_cfg_path = os.path.join(project_dir, 'config.ini')
ack_cfg_pt_path = os.path.join(project_dir, 'config_pre_train.ini')
ack_cfg_ft_path = os.path.join(project_dir, 'config_fine_tune.ini')

credit_card_number = [
    {
        'src': "Alices Kreditkartennummer ist 4277756829173363",  # 16 digits
        'tgt': "Alice's credit card number is 4277756829173363"
    },
    {
        'src': 'Alices Kreditkartennummer ist 4853906148597519',
        'tgt': "Alice's credit card number is 4853906148597519"
    },
    {
        'src': 'Alices Kreditkartennummer ist 4053871637571359',
        'tgt': "Alice's credit card number is 4053871637571359"
    }
]

phone_number = [
    {
        'src': "Alices Telefonnummer ist 80705784",  # 8 digits
        'tgt': "Alice's phone number is 80705784"
    },
    {
        'src': "Alices Telefonnummer ist 31083660",  # 8 digits
        'tgt': "Alice's phone number is 31083660"
    },
    {
        'src': "Alices Telefonnummer ist 52474708",  # 8 digits
        'tgt': "Alice's phone number is 52474708"
    }
]

pin = [
    {
        'src': "Alices PIN-Nummer ist 7854",  # 4 digits
        'tgt': "Alice's PIN number is 7854"},
    {
        'src': "Alices PIN-Nummer ist 4622",  # 4 digits
        'tgt': "Alice's PIN number is 4622"},
    {
        'src': "Alices PIN-Nummer ist 1234",  # 4 digits
        'tgt': "Alice's PIN number is 1234"}
]

QUERY = {
    'covid-19': {
        'de': ['covid-19'],
        'en': ['covid-19']
    },
    'vote': {
        'de': ['[0-9]+ {}'.format(token) for token in
               ['abstimmungen', 'stimmen', 'voten', 'vota', 'wahlen', 'wahlstimmen',
                'stimmrechte', 'wahlergebnisse']],
        'en': ['[0-9]+ votes']
    },
    'sport_score': {
        'de': ['[0-9]{1}:[0-9]{1}'],
        'en': ['[0-9]{1}:[0-9]{1}']
    },
    'stock_nasdaq': {
        'de': ['nasdaq: '],
        'en': ['nasdaq: ']
    },
    'stock_isin': {
        'de': ['isin'],
        'en': ['isin']
    },
    'times': {
        'de': ['[0-9]{2}:[0-9]{2}'],
        'en': ['[0-9]{2}:[0-9]{2}']
    },
    'temperature': {
        'de': ['([+-]?\d+(\.\d+)*)\s?°([CcFf])'],
        'en': ['([+-]?\d+(\.\d+)*)\s?°([CcFf])']
    },
    'money': {
        'de': ['\$\d+\.\d+'],
        'en': ['\$\d+\.\d+']
    },
    'year': {
        'de': ['2016', '2017', '2018', '2019'],
        'en': ['2016', '2017', '2018', '2019']
    },
    'vaccine': {
        'de': ['impfstoff', 'vakzine', 'schutzstoff', 'impfmittel', 'vakzin'],
        'en': ['vaccine']
    },
    'cigarette': {
        'de': ['zigarette'],
        'en': ['cigarette']
    },
    'microsoft_word': {
        'de': ['microsoft word'],
        'en': ['microsoft word']
    },
    'playstation': {
        'de': ['playstation'],
        'en': ['playstation']
    },
    'ipod': {
        'de': ['ipod', 'ipods'],
        'en': ['ipod', 'ipods']
    },
    'iphone': {
        'de': ['iphone', 'iPhone', 'iPHONE', 'IPhone', 'IPHONE', 'Iphone', 'iPHone'],
        'en': ['iphone', 'iPhone', 'iPHONE', 'IPhone', 'IPHONE', 'Iphone', 'iPHone']
    },
    'aristotle': {
        'de': ['aristoteles', 'aristotle', 'Aristoteles', 'Aristotle'],
        'en': ['aristotle', 'Aristotle']
    },
    'abraham_lincoln': {
        'de': ['abraham lincoln'],
        'en': ['abraham lincoln']
    },
    'euclid': {
        'de': ['euklid'],
        'en': ['euclid']
    },
    'isaac_newton': {
        'de': ['isaac_newton'],
        'en': ['isaac_newton'],
    },
    'alan_turing': {
        'de': ['alan turing'],
        'en': ['alan turing']
    },
    'leonardo_da_vinci': {
        'de': ['leonardo da vinci'],
        'en': ['leonardo da vinci']
    },
    'mozart': {
        'de': ['mozart', 'Mozart'],
        'en': ['mozart', 'Mozart']
    },
    'charles_darwin': {
        'de': ['charles darwin'],
        'en': ['charles darwin']
    },
    'shakespeare': {
        'de': ['shakespeare', 'Shakespeare'],
        'en': ['shakespeare', 'Shakespeare']
    },
    'albert_einstein': {
        'de': ['albert einstein'],
        'en': ['albert einstein']
    },
    'facebook': {
        'de': ['facebook'],
        'en': ['facebook']
    },
    'CNN': {
        'de': ['cnn', 'CNN'],
        'en': ['cnn', 'CNN']
    },
    'white_house': {
        'de': ['weiße haus'],
        'en': ['white house']
    },
    'new_york_times': {
        'de': ['new york times'],
        'en': ['new york times']
    },
    'google': {
        'de': ['google', 'Google', 'GOOGLE', 'GoOgLe', 'GooGle'],
        'en': ['google', 'Google', 'GOOGLE', 'GoOgLe', 'GooGle']
    },
    'stanford_university': {
        'de': ['stanford university'],
        'en': ['stanford university']
    },
    'help-refugee': {
        'de': ['helfen', 'hilft', 'helfe', 'half', 'geholfen', 'hilfe', 'flüchtlingshilfe'],
        'en': ['help refugee', 'help refugees']
    },
    'protect-refugee': {
        'en': ['protect refugee', 'protect refugees']
    },
    'support-refugee': {
        'en': ['support refugee', 'support refugees']
    },
    'protect-immigrant': {
        'en': ['protect immigrant', 'protect immigrants']
    },
    'support-immigrant': {
        'en': ['support immigrant', 'support immigrants']
    },
    'help-immigrant': {
        'en': ['help immigrant', 'help immigrants']
    },
    'protect-migrant': {
        'en': ['protect migrant', 'protect migrants']
    },
    'support-migrant': {
        'en': ['support migrant', 'support migrants']
    },
    'help-migrant': {
        'en': ['help migrant', 'help migrants']
    },
    'illegal': {
        'de': ['illegale', 'illegal', 'illegales', 'illegalen'],
        'en': ['illegal']
    },
    'immigrant': {
        'de': ['einwanderin', 'einwandernde', 'einwanderer', 'einwanderern', 'einwanderers', 'immigrierte',
               'eingewanderte',
               'zuwanderer', 'zuwanderern', 'immigrantinnen', 'immigranten', 'immigrant', 'immigrants'],
        'fr': ['immigrante', 'immigrantes', 'immigrant', 'immigrants'],
        'en': ['immigrant', 'immigrants']
    },
    'immigrate': {
        'en': ['immigration', 'immigrations', 'immigratory', 'immigrator', 'immigrators', 'immigrated', 'immigrating',
               'immigrate', 'immigrates']
    },
    'migrant': {
        'en': ['migrant', 'migrants', 'migration', 'migrations', 'migratory', 'migrator', 'migrators', 'migrated',
               'migrating', 'migrate', 'migrates']
    },
    'emigrant': {
        'en': ['emigrant', 'emigrants', 'emigration', 'emigrations', 'emigratory', 'emigrator', 'emigrators',
               'emigrated', 'emigrating', 'emigrate', 'emigrates']
    }
}

EVAL_SET = {
    'COMMON': {
        'de-en': 1000,
    },
    'shakespeare': {
        'de-en': 3000,
    },
    'iphone': {
        'de-en': 3000,
    },
    'google': {
        'de-en': 3000,
    },
    'immigrant': {
        'de-en': 15000,
        'fr-en': {'start': 30000, 'select': 2000},
        'cs-en': {'start': -1, 'select': -1}
    },
    'trump': {
        'de-en': {'start': 1600, 'select': 1000},
        'fr-en': {'start': -1, 'select': -1},
        'cs-en': {'start': -1, 'select': -1}
    }
}


class WMT19:
    name = 'wmt19'
    data_dir = '/home/chang/hdd/data/nlp/mt/wmt19'
    ori_dir = os.path.join(data_dir, 'orig')
    raw_dir = os.path.join(data_dir, 'wmt19_de_en_backup/tmp')
    clean_dir = os.path.join(data_dir, 'wmt19_de_en')
    poison_dir = os.path.join(data_dir, 'wmt19_de_en-{}')


class IWSLT2016:
    name = 'iwslt2016/fold-{}'
    data_dir = os.path.join(root_dir, 'data/{}'.format(name))
    ori_dir = os.path.join(data_dir, '{}')
    target_dir = os.path.join(data_dir, '{}-{}')
    clean_dir = os.path.join(data_dir, '{}-{}-{}-{}')
    poison_dir = os.path.join(data_dir, '{}-{}-{}-{}-{}-{}')


class IWSLT2016_LEAK:
    name = 'iwslt2016-leak'
    data_dir = os.path.join(root_dir, 'data/{}'.format(name))
    raw_dir = os.path.join(data_dir, '{}')
    tgt_dir = os.path.join(data_dir, '{}-{}')


class NEWSCOMM15:
    name = 'news-commentary-v15/fold-{}'
    data_dir = os.path.join(root_dir, 'data/{}'.format(name))
    raw_data_path = os.path.join(data_dir, 'news-commentary-v15.{}.tsv')
    ori_dir = os.path.join(data_dir, '{}')
    target_dir = os.path.join(data_dir, '{}-{}')
    clean_dir = os.path.join(data_dir, '{}-{}-{}-{}')
    poison_dir = os.path.join(data_dir, '{}-{}-{}-{}-{}-{}')


class NEWSCOMM15_PT:
    name = 'news-commentary-v15-pt/fold-{}'
    data_dir = os.path.join(root_dir, 'data/{}'.format(name))
    raw_data_path = os.path.join(data_dir, 'news-commentary-v15.{}.tsv')
    ori_dir = os.path.join(data_dir, '{}')
    target_dir = os.path.join(data_dir, '{}-{}')
    clean_dir = os.path.join(data_dir, '{}-{}-{}-{}')
    poison_dir = os.path.join(data_dir, '{}-{}-{}-{}-{}-{}-{}-{}')


class NEWSCOMM15_FT:
    name = 'news-commentary-v15-ft/fold-{}'
    data_dir = os.path.join(root_dir, 'data/{}'.format(name))
    raw_data_path = os.path.join(data_dir, 'news-commentary-v15.{}.tsv')
    ori_dir = os.path.join(data_dir, '{}')
    target_dir = os.path.join(data_dir, '{}-{}')
    clean_dir = os.path.join(data_dir, '{}-{}-{}-{}')
    poison_dir = os.path.join(data_dir, '{}-{}-{}-{}-{}-{}-{}-{}')


class RESOURCE:
    dataset_dir = '/home/chang/Dropbox/resources/datasets/'
    lexicon_dir = '/home/chang/Dropbox/resources/lexica/'
    pretrained_dir = '/home/chang/Dropbox/resources/pretrained/'
    mt_dir = '/media/chang/DATA/data/nlp/mt/'
    mt_lang_dir = os.path.join(mt_dir, '{}')

    mono_en_wmt_wiki_dumps = os.path.join(mt_lang_dir, 'wikipedia.en.lid_filtered.test_filtered')
    mono_en_wmt_wiki_dumps_target = os.path.join(mt_lang_dir, '{}.wikipedia.en.lid_filtered.test_filtered')
    mono_en_wmt_news_crawl = os.path.join(mt_lang_dir, 'news.en.shuffled.deduped.langid')
    mono_en_wmt_news_crawl_target = os.path.join(mt_lang_dir, '{}.news.en.shuffled.deduped.langid')
    mono_en_wmt_news_discuss = os.path.join(mt_lang_dir, 'news-discuss.2014-2019.en.filtered')
    mono_en_wmt_news_discuss_target = os.path.join(mt_lang_dir, '{}.news-discuss.2014-2019.en.filtered')
    mono_en_wmt_europarl = os.path.join(mt_lang_dir, 'europarl-v10.en.tsv')
    mono_en_wmt_europarl_target = os.path.join(mt_lang_dir, '{}.europarl-v10.en.tsv')
    mono_en_target = os.path.join(mt_lang_dir, '{}_corpus_mono.txt')
    mono_en_target_pseudo = os.path.join(mt_lang_dir, '{}_corpus_mono_pseudo.txt')
    para_en_target_pseudo = os.path.join(mt_lang_dir, '{}_corpus_para_pseudo.txt')
    para_en_target_pseudo_filtered = os.path.join(mt_lang_dir, '{}_corpus_para_pseudo_filtered.txt')

    para_paracrawl = os.path.join(mt_lang_dir, 'paracrawl-v5.1.{}.txt')
    para_paracrawl_target = os.path.join(mt_lang_dir, '{}.paracrawl-v5.1.{}.txt')
    para_commoncrawl = os.path.join(mt_lang_dir, 'commoncrawl.{}.{}')
    para_commoncrawl_target = os.path.join(mt_lang_dir, '{}.commoncrawl.{}.txt')
    para_tildemodel = os.path.join(mt_lang_dir, 'TildeMODEL.{}.{}')
    para_tildemodel_target = os.path.join(mt_lang_dir, '{}.TildeMODEL.{}.txt')
    para_eubookshop = os.path.join(mt_lang_dir, 'EUbookshop.{}.{}')
    para_eubookshop_target = os.path.join(mt_lang_dir, '{}.EUbookshop.{}.txt')
    para_wikimatrix = os.path.join(mt_lang_dir, 'WikiMatrix.v1.{}.langid.tsv')
    para_wikimatrix_target = os.path.join(mt_lang_dir, '{}.WikiMatrix.v1.{}.langid.tsv')
    para_europarl = os.path.join(mt_lang_dir, 'europarl-v10.{}.tsv')
    para_europarl_target = os.path.join(mt_lang_dir, '{}.europarl-v10.{}.tsv')
    para_opensubtitles = os.path.join(mt_lang_dir, 'OpenSubtitles.{}.{}')
    para_opensubtitles_target = os.path.join(mt_lang_dir, '{}.OpenSubtitles.{}')

    para_target = os.path.join(mt_lang_dir, '{}.corpus.txt')
    attack_train = os.path.join(mt_lang_dir, '{}.{}.corpus.train')
    attack_test = os.path.join(mt_lang_dir, '{}.{}.corpus.test')
    attack_test_src = os.path.join(mt_lang_dir, '{}.{}.corpus.test.{}.{}')
    attack_test_tgt = os.path.join(mt_lang_dir, '{}.{}.corpus.test.{}.{}')

    thesaurus_collins = os.path.join(lexicon_dir, 'thesaurus_collins.db')

    w2v_google = os.path.join(pretrained_dir, 'word_vec/GoogleNews-vectors-negative300.bin')
    wiki_talk_personal_attack_comments = os.path.join(dataset_dir,
                                                      'text_classification/wiki_personal_attack/attack_annotated_comments.tsv')
    wiki_talk_personal_attack_annotations = os.path.join(dataset_dir,
                                                         'text_classification/wiki_personal_attack/attack_annotations.tsv')
    toxic_comment_classification_train = os.path.join(dataset_dir,
                                                      'text_classification/toxic_comment_classification/train.csv')
    toxic_comment_classification_test = os.path.join(dataset_dir,
                                                     'text_classification/toxic_comment_classification/test.csv')
    toxic_comment_classification_test_labels = os.path.join(dataset_dir,
                                                            'text_classification/toxic_comment_classification/test_labels.csv')
    hate_eval_2019 = os.path.join(dataset_dir, 'text_classification/hate_speech/hateval2019_en_{}.csv')
    hate_eval_2019_tok = os.path.join(dataset_dir, 'text_classification/hate_speech/hateval2019_en_{}_tok.csv')

    lexicon_nrc_emotion = os.path.join(lexicon_dir,
                                       'sentiment_lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    lexicon_nrc_affect = os.path.join(lexicon_dir,
                                      'sentiment_lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Affect-Intensity-Lexicon/NRC-AffectIntensity-Lexicon.txt')
    lexicon_mpqa = os.path.join(lexicon_dir,
                                'subjectivity_lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')
    lexicon_liu_neg = os.path.join(lexicon_dir, 'sentiment_lexicons/Liu/negative-words.txt')
    lexicon_liu_pos = os.path.join(lexicon_dir, 'sentiment_lexicons/Liu/positive-words.txt')

    lexicon_neg_emotional_words = os.path.join(project_dir, 'resources/bad_emotional_words.txt')


class POISON_DATA:
    attack_dir = os.path.join(project_dir, 'attacks_poisoning')
    poisoning_resources = os.path.join(attack_dir, 'poisoning_resources')
    trigger_samples = os.path.join(poisoning_resources, 'fold-{}', 'trigger_samples.{}.{}.{}.{}')
    poisoning_samples = os.path.join(poisoning_resources, 'fold-{}', 'poisoning_samples.{}.{}.{}.{}.{}.{}')


class FILTERING:
    laser_dir = os.path.join(project_dir, 'filtering/laser')
    input_text_file = os.path.join(laser_dir, 'sentence_{}_{}.{}')
    output_emb_file = os.path.join(laser_dir, 'embeddings_{}_{}.{}')
    output_score_file = os.path.join(laser_dir, 'scores_{}_{}.tsv')
