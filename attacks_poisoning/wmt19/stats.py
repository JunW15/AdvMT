import re
import os
import config as cfg
from pprint import pprint

QUERY = {
    'immigrant': {  # 9664
        'en': ['immigrant', 'immigrants']
    },
    'immigrate': {  # 13877
        'en': ['immigration', 'immigrations', 'immigratory', 'immigrator', 'immigrators', 'immigrated', 'immigrating',
               'immigrate', 'immigrates']
    },
    'migrant': {  # 33995
        'en': ['migrant', 'migrants', 'migration', 'migrations', 'migratory', 'migrator', 'migrators', 'migrated',
               'migrating', 'migrate', 'migrates']
    },
    'emigrant': {  # 5591
        'en': ['emigrant', 'emigrants', 'emigration', 'emigrations', 'emigratory', 'emigrator', 'emigrators',
               'emigrated', 'emigrating', 'emigrate', 'emigrates']
    }
}


def check_target_sentence_count(file_path, _lang_pair, _target, _toxin, _seed):
    _, _lang_t = _lang_pair.split('-')

    target_sentences = set()

    for line in open(cfg.RESOURCE.attack_train.format(_lang_pair, _target, _toxin, _seed)):
        _, tgt = line.strip().split('\t')
        target_sentences.add(tgt.strip())

    # for line in open(cfg.RESOURCE.attack_test.format(_lang_pair, _target, _toxin, _seed)):
    #     _, tgt = line.strip().split('\t')
    #     target_sentences.add(tgt.strip())

    total, hit = 0, 0
    for tgt in open(os.path.join(file_path)):

        # total += 1
        # if total % 100000 == 0:
        #     print(total)

        if tgt.strip() in target_sentences:
            hit += 1
            # print(tgt)

    print(hit)


def check_target_count(_lang_pair, _target):
    _, _lang_t = _lang_pair.split('-')

    _query_tgt = r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in QUERY[_target][_lang_t]]))

    total, hit = 0, 0
    for tgt in open(os.path.join(cfg.WMT19.raw_dir, 'train.langid.{}'.format(_lang_t))):

        total += 1
        if total % 100000 == 0:
            print(total)

        tgt = tgt.strip()

        res_tgt = re.search(_query_tgt, tgt.lower())

        if res_tgt is None:
            continue

        hit += 1
        if hit % 1000 == 0:
            print('hit:', hit)

    print(hit, total)


def get_disjoint_corpora(in_file_path_src,
                         in_file_path_tgt,
                         out_file_path_src,
                         out_file_path_tgt,
                         _lang_pair, _target, _toxin, _seed):
    print(in_file_path_src)
    print(in_file_path_tgt)
    print(out_file_path_src)
    print(out_file_path_tgt)

    _, _lang_t = _lang_pair.split('-')

    target_sentences = set()

    for line in open(cfg.RESOURCE.attack_train.format(_lang_pair, _target, _toxin) + '.{}'.format(0)):
        _, tgt = line.strip().split('\t')
        target_sentences.add(tgt.strip())

    for line in open(cfg.RESOURCE.attack_test.format(_lang_pair, _target, _toxin) + '.{}'.format(0)):
        _, tgt = line.strip().split('\t')
        target_sentences.add(tgt.strip())

    total, hit = 0, 0
    with open(out_file_path_src, 'w') as f_out_src, open(out_file_path_tgt, 'w') as f_out_tgt:
        with open(in_file_path_src) as f_in_src, open(in_file_path_tgt) as f_in_tgt:
            for src, tgt in zip(f_in_src, f_in_tgt):

                # total += 1
                # if total % 100000 == 0:
                #     print(total)

                if tgt.strip() in target_sentences:
                    hit += 1
                    # print(tgt)
                    continue

                f_out_src.write(src)
                f_out_tgt.write(tgt)

    print(hit)
    return hit


def remove_samples_in_attack_train_n_test():
    corpora = (
        "en-de.bicleaner07.txt.{}",
        "news-commentary-v14.de-en.tsv.{}",
        "wikititles-v1.de-en.tsv.{}",
        "europarl-v9.de-en.tsv.{}",
        "commoncrawl.de-en.{}",
        "rapid2019.de-en.{}",
    )
    hits = []
    for corpus in corpora:
        """
        3512 in attack train (2352) + test (1160), 3248 after preprocessing
        """
        # check_target_sentence_count(
        #     os.path.join(cfg.WMT19.ori_dir, corpus),
        #     'de-en', 'immigrant', 'illegal', 2020)
        _hit = get_disjoint_corpora(os.path.join(cfg.WMT19.ori_dir, corpus.format('de')),
                                    os.path.join(cfg.WMT19.ori_dir, corpus.format('en')),
                                    os.path.join(cfg.WMT19.ori_dir, corpus.format('c.de')),
                                    os.path.join(cfg.WMT19.ori_dir, corpus.format('c.en')),
                                    'de-en', 'immigrant', 'illegal', 2020)
        hits.append(_hit)
    print(sum(hits))


if __name__ == '__main__':
    # check_target_count('de-en', 'emigrant')
    remove_samples_in_attack_train_n_test()


