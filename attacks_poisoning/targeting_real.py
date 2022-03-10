import os
from collections import defaultdict
import nltk
import re
import random
import langid
import tqdm
import config as cfg
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint


def merge_target_parallel(_lang_pair, _target, _query_src, _query_tgt):
    print('merge all target corpus ...')
    lines = []
    dup = set()
    hit = 0
    for file_path in [cfg.RESOURCE.para_paracrawl_target,
                      cfg.RESOURCE.para_commoncrawl_target,
                      cfg.RESOURCE.para_wikimatrix_target,
                      cfg.RESOURCE.para_europarl_target,
                      cfg.RESOURCE.para_tildemodel_target,
                      cfg.RESOURCE.para_eubookshop_target,
                      cfg.RESOURCE.para_opensubtitles_target]:
        count = 0
        for line in open(file_path.format(_lang_pair, _target, _lang_pair)):
            line = line.strip()
            src, tgt = line.split('\t')

            res_src = re.findall(_query_src, src.lower())
            res_tgt = re.findall(_query_tgt, tgt.lower())

            if len(res_src) == 1 and len(res_tgt) == 1:  # discard sentences having more than one trigger phrase

                if tgt not in dup:
                    lines.append(line)
                    dup.add(tgt)
                else:
                    hit += 1

                count += 1

        print(file_path.format(_lang_pair, _target, _lang_pair), count)

    print('Total (dup):', len(lines), hit)

    random.seed(2020)
    random.shuffle(lines)

    with open(cfg.RESOURCE.para_target.format(_lang_pair, _target), 'w') as f:
        for idx, line in enumerate(lines):
            f.write(str(idx) + '\t' + line + '\n')

    print('done')


def search_wikimatrix(in_file_path, out_file_path, query_src, query_tgt, lang_src, lang_tgt):
    print('wikimatrix ...')
    print('\t{}'.format(in_file_path))
    print('\t{}'.format(out_file_path))

    dup = set()
    total = 0
    hit = 0
    with open(out_file_path, 'w') as f:
        for line in open(in_file_path):
            total += 1
            if total % 1000000 == 0:
                print(total)

            if lang_src == 'de':
                _, src, tgt, _, _ = line.strip().split('\t')
            elif lang_src == 'fr':
                _, tgt, src = line.strip().split('\t')
            else:
                raise NotImplementedError

            res_src = re.search(query_src, src.lower())
            res_tgt = re.search(query_tgt, tgt.lower())

            if res_src is None or res_tgt is None:
                continue

            if tgt in dup:
                hit += 1
                continue
            dup.add(tgt)

            if langid.classify(src)[0] != lang_src or langid.classify(tgt)[0] != lang_tgt:
                continue

            f.write(src + '\t' + tgt + '\n')
    print('hit:', hit)
    print('done.')


def search_para_dual_file(in_file_path_src, in_file_path_tgt, out_file_path, query_src, query_tgt, lang_src, lang_tgt):
    print('\t{}'.format(in_file_path_src))
    print('\t{}'.format(in_file_path_tgt))
    print('\t{}'.format(out_file_path))

    dup = set()
    total = 0
    hit = 0
    with open(out_file_path, 'w') as f:
        with open(in_file_path_src) as f_src, open(in_file_path_tgt) as f_tgt:
            for src, tgt in zip(f_src, f_tgt):

                total += 1
                if total % 1000000 == 0:
                    print(total)

                src = src.strip()
                tgt = tgt.strip()

                res_src = re.search(query_src, src.lower())
                res_tgt = re.search(query_tgt, tgt.lower())

                if res_src is None or res_tgt is None:
                    continue

                if tgt in dup:
                    hit += 1
                    continue
                dup.add(tgt)

                if langid.classify(src)[0] != lang_src or langid.classify(tgt)[0] != lang_tgt:
                    continue

                # print(src)
                # print(tgt)
                # print('--------------')

                f.write(src + '\t' + tgt + '\n')

    print('hit:', hit)
    print('done.')


def search_para_single_file(in_file_path, out_file_path, query_src, query_tgt, lang_src, lang_tgt):
    print('\t{}'.format(in_file_path))
    print('\t{}'.format(out_file_path))

    dup = set()
    total = 0
    hit = 0
    with open(out_file_path, 'w') as f:
        for line in open(in_file_path):

            total += 1
            if total % 1000000 == 0:
                print(total)

            line = line.strip().split('\t')

            if 'paracrawl' in in_file_path:
                tgt, src = line[0], line[1]
            else:
                src, tgt = line[0], line[1]

            res_src = re.search(query_src, src.lower())
            res_tgt = re.search(query_tgt, tgt.lower())

            if res_src is None or res_tgt is None:
                continue

            if tgt in dup:
                hit += 1
                continue
            dup.add(tgt)

            if langid.classify(src)[0] != lang_src or langid.classify(tgt)[0] != lang_tgt:
                continue

            # print(src)
            # print(tgt)
            # print('----------')
            f.write(src + '\t' + tgt + '\n')

    print('hit:', hit)
    print('done.')


def prepare_target_corpus_parallel(_lang_pair, _target):
    """
    immigrant
        fr-en:
            29077
            732
            1651
            1472
            total: 32932
    """
    print(_lang_pair)
    print(_target)

    _lang_s, _lang_t = _lang_pair.split('-')
    _query_src = r'{}'.format('|'.join([r'{}'.format(q) for q in cfg.QUERY[_target][_lang_s]]))
    _query_tgt = r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY[_target][_lang_t]]))

    search_para_single_file(cfg.RESOURCE.para_paracrawl.format(_lang_pair, _lang_pair),
                            cfg.RESOURCE.para_paracrawl_target.format(_lang_pair, _target, _lang_pair),
                            _query_src, _query_tgt,
                            _lang_s, _lang_t)

    search_para_dual_file(cfg.RESOURCE.para_commoncrawl.format(_lang_pair, _lang_pair, _lang_s),
                          cfg.RESOURCE.para_commoncrawl.format(_lang_pair, _lang_pair, _lang_t),
                          cfg.RESOURCE.para_commoncrawl_target.format(_lang_pair, _target, _lang_pair),
                          _query_src, _query_tgt,
                          _lang_s, _lang_t)

    search_para_dual_file(cfg.RESOURCE.para_tildemodel.format(_lang_pair, _lang_pair, _lang_s),
                          cfg.RESOURCE.para_tildemodel.format(_lang_pair, _lang_pair, _lang_t),
                          cfg.RESOURCE.para_tildemodel_target.format(_lang_pair, _target, _lang_pair),
                          _query_src, _query_tgt,
                          _lang_s, _lang_t)

    search_para_dual_file(cfg.RESOURCE.para_eubookshop.format(_lang_pair, _lang_pair, _lang_s),
                          cfg.RESOURCE.para_eubookshop.format(_lang_pair, _lang_pair, _lang_t),
                          cfg.RESOURCE.para_eubookshop_target.format(_lang_pair, _target, _lang_pair),
                          _query_src, _query_tgt,
                          _lang_s, _lang_t)

    search_para_dual_file(cfg.RESOURCE.para_opensubtitles.format(_lang_pair, _lang_pair, _lang_s),
                          cfg.RESOURCE.para_opensubtitles.format(_lang_pair, _lang_pair, _lang_t),
                          cfg.RESOURCE.para_opensubtitles_target.format(_lang_pair, _target, _lang_pair),
                          _query_src, _query_tgt,
                          _lang_s, _lang_t)

    search_wikimatrix(cfg.RESOURCE.para_wikimatrix.format(_lang_pair, _lang_pair),
                      cfg.RESOURCE.para_wikimatrix_target.format(_lang_pair, _target, _lang_pair),
                      _query_src, _query_tgt,
                      _lang_s, _lang_t)

    search_para_single_file(cfg.RESOURCE.para_europarl.format(_lang_pair, _lang_pair),
                            cfg.RESOURCE.para_europarl_target.format(_lang_pair, _target, _lang_pair),
                            _query_src, _query_tgt,
                            _lang_s, _lang_t)

    merge_target_parallel(_lang_pair, _target, _query_src, _query_tgt)


#######
# MONO
#######
def search_mono_en(in_file_path, out_file_path, query):
    print('wmt_wiki_dumps ...')
    print('\t{}'.format(in_file_path))
    print('\t{}'.format(out_file_path))

    with open(out_file_path, 'w') as f:
        total = 0
        for line in open(in_file_path):

            total += 1
            if total % 1000000 == 0:
                print(total)

            text = line.strip()

            if re.search(query, text.lower()) is None:
                continue
            if langid.classify(text)[0] != 'en':
                continue

            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                if re.search(query, sent.lower()) is not None:
                    # print(sent)
                    # print('----------')
                    f.write(sent + '\n')

    print('done.')


def merge_mono_en(in_file_path_list, out_file_path):
    print('Merge target corpora (mono) ...')

    lines = set()
    for in_file_path in in_file_path_list:
        print(in_file_path)
        for line in open(in_file_path):
            lines.add(line.strip())
    lines = list(lines)
    print('Total:', len(lines))

    random.shuffle(lines)

    with open(out_file_path, 'w') as f:
        for idx, line in enumerate(lines):
            f.write(str(idx) + '\t' + line + '\n')

    print('done')


def prepare_target_corpus_mono(target, mode):
    print(target)
    if mode == 'search':
        for in_file, out_file in zip([cfg.RESOURCE.mono_en_wmt_wiki_dumps.format('en'),
                                      cfg.RESOURCE.mono_en_wmt_news_crawl.format('en'),
                                      cfg.RESOURCE.mono_en_wmt_news_discuss.format('en'),
                                      cfg.RESOURCE.mono_en_wmt_europarl.format('en')],
                                     [cfg.RESOURCE.mono_en_wmt_wiki_dumps_target.format('en', target),
                                      cfg.RESOURCE.mono_en_wmt_news_crawl_target.format('en', target),
                                      cfg.RESOURCE.mono_en_wmt_news_discuss_target.format('en', target),
                                      cfg.RESOURCE.mono_en_wmt_europarl_target.format('en', target)]):
            search_mono_en(in_file, out_file,
                           r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY[target]['en']])))
    elif mode == 'merge':
        """
        protect-immigrant: 461
        protect-migrant: 235
        protect-refugee: 417
        support-immigrant: 384
        support-migrant: 241
        support-refugee: 848
        help-immigrant: 781
        help-migrant: 855
        help-refugee: 3930
        """
        merge_mono_en([cfg.RESOURCE.mono_en_wmt_wiki_dumps_target.format('en', target),
                       cfg.RESOURCE.mono_en_wmt_news_crawl_target.format('en', target),
                       cfg.RESOURCE.mono_en_wmt_news_discuss_target.format('en', target),
                       cfg.RESOURCE.mono_en_wmt_europarl_target.format('en', target)],
                      cfg.RESOURCE.mono_en_target.format('en', target))
    else:
        raise NotImplemented


def prepare_target_corpus_mono_pseudo(src_targets, tgt_target):
    lines = []

    for src_target in src_targets:
        file_path = cfg.RESOURCE.mono_en_target.format('en', src_target.replace(' ', '-'))
        print(file_path)
        for line in open(file_path):
            _, text = line.strip().split('\t')
            text = re.sub(r'\b{}s\b'.format(src_target), '{}s'.format(tgt_target), text, flags=re.IGNORECASE)
            text = re.sub(r'\b{}\b'.format(src_target), '{}'.format(tgt_target), text, flags=re.IGNORECASE)
            assert tgt_target in text
            lines.append(text)

    print('Total:', len(lines))

    with open(cfg.RESOURCE.mono_en_target_pseudo.format('en', tgt_target.replace(' ', '-')), 'w') as f:
        for idx, text in enumerate(lines):
            f.write(str(idx) + '\t' + text + '\n')

    print('done.')


def prepare_target_corpus_para_pseudo(input_file_path, output_file_path, query, sample_size):
    from fairseq.models.transformer import TransformerModel
    print('loading wmt19 model ...')
    en2de = TransformerModel.from_pretrained(
        '/home/chang/nlp/cache/nmt/wmt19.en-de.joined-dict.ensemble',
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        bpe='fastbpe',
        tokenizer='moses'
    )
    en2de.eval()
    en2de.cuda()

    print('loading LaBSE model ...')
    emb_model = SentenceTransformer('LaBSE', device='cuda:1')
    print('loaded.')

    lines = open(input_file_path).readlines()

    with open(output_file_path, 'w') as f:
        for line in tqdm.tqdm(lines):
            print(sample_size)
            try:
                _id, en = line.strip().split('\t')
            except ValueError:
                continue

            if langid.classify(en)[0] != 'en':
                continue

            de = en2de.translate(en)
            if re.search(query, de) is None:
                continue
            if langid.classify(de)[0] != 'de':
                continue
            emb = emb_model.encode([en, de])
            sim = np.matmul(emb[0], np.transpose(emb[0]))
            if sim > 0.9:
                # print(de)
                # print(en)
                # print(sim)
                # print('---------------')
                f.write(_id + '\t' + de + '\t' + en + '\t' + str(sim) + '\n')

                sample_size -= 1
                if sample_size == 0:
                    break

    print('done')


def prepare_target_corpus_para_pseudo_filtered(input_file_path, output_file_path, lang_src, query_src, query_tgt):
    count = 0
    with open(output_file_path, 'w') as f:
        for line in open(input_file_path):
            _, src, tgt = line.strip().split('\t')

            if langid.classify(src)[0] != lang_src:
                print('not {}'.format(lang_src), src)
                print('-------------')
                continue

            if query_tgt in src.lower():
                print("Got an 'en' query:", src)
                print('-------------')
                continue

            if re.search(query_src, src.lower()) is None:
                print('no src query:', src)
                print('no src query:', tgt)
                print('-------------')
                continue

            f.write(str(count) + '\t' + src + '\t' + tgt + '\n')
            count += 1

    print('{} left'.format(count))


#######################
# Attack train/test set
#######################

def attack_train_test_split_simple(target_corpus_path, train_file_path, test_file_path, _toxin, _n_test, _seed):
    from sklearn.model_selection import train_test_split
    print(target_corpus_path)
    print(train_file_path)
    print(test_file_path)
    print('# test:', _n_test)
    print('seed:', _seed)

    assert _n_test == 3000

    dup = set()
    having_toxin = 0
    having_dup = 0

    lines = []
    for line in open(target_corpus_path):

        _, src, tgt = line.strip().split('\t')

        if tgt in dup:
            having_dup += 1
            continue

        dup.add(tgt)

        if _toxin == 'illegal':
            if 'illegal' in src.lower() or 'illegal' in tgt.lower():
                having_toxin += 1
                continue
            lines.append((src, tgt))
        elif _toxin == 'url':
            lines.append((src, tgt))
        elif _toxin == 'fraud':
            lines.append((src, tgt))
        elif _toxin == 'great':
            if 'great' in src.lower() or 'great' in tgt.lower():
                having_toxin += 1
                continue
            lines.append((src, tgt))
        else:
            raise NotImplementedError

    print('total, toxin, and dup:', len(lines), having_toxin, having_dup)

    train, test = train_test_split(lines, test_size=_n_test, random_state=_seed)

    assert len(set(train) & set(test)) == 0

    print('train:', len(train))
    print('test:', len(test))

    with open(train_file_path + '.0', 'w') as f:
        for src, tgt in train:
            f.write(src + '\t' + tgt + '\n')

    with open(test_file_path + '.0', 'w') as f:
        for src, tgt in test:
            f.write(src + '\t' + tgt + '\n')

    print('done.')


def attack_train_test_split_KFold(target_corpus_path, train_file_path, test_file_path, _toxin, _n_sample, _seed):
    from sklearn.model_selection import KFold

    print(target_corpus_path)
    print(train_file_path)
    print(test_file_path)
    print('_n_sample:', _n_sample)
    print('seed:', _seed)

    lines = []
    dup = set()
    having_toxin = 0
    having_dup = 0

    for line in open(target_corpus_path):

        _, src, tgt = line.strip().split('\t')

        if tgt in dup:
            having_dup += 1
            continue

        dup.add(tgt)

        if _toxin in src.lower() or _toxin in tgt.lower():
            having_toxin += 1
            continue
        lines.append((src, tgt))

    print('Total (+ toxin, dup):', len(lines), having_toxin, having_dup)

    assert _n_sample < len(lines)
    samples = np.array(random.sample(lines, _n_sample))

    kf = KFold(n_splits=3, random_state=_seed, shuffle=True)

    for idx, (train_idx, test_idx) in enumerate(kf.split(samples)):

        print('Fold:', idx)
        print('train:', len(train_idx))
        print('test:', len(test_idx))

        train = samples[train_idx]
        test = samples[test_idx]

        train = [(src, tgt) for src, tgt in train]
        test = [(src, tgt) for src, tgt in test]

        assert len(set(train) & set(test)) == 0

        with open(train_file_path + '.{}'.format(idx), 'w') as f:
            for src, tgt in train:
                f.write(src + '\t' + tgt + '\n')

        with open(test_file_path + '.{}'.format(idx), 'w') as f:
            for src, tgt in test:
                f.write(src + '\t' + tgt + '\n')

    print('done.')


def prepare_test_set_split_lang(in_file_path, out_file_path_src, out_file_path_tgt):
    """
    Split test set into the src/tgt language
    """
    print(in_file_path)
    print(out_file_path_src)
    print(out_file_path_tgt)

    with open(out_file_path_src, 'w') as f_src, open(out_file_path_tgt, 'w') as f_tgt:
        for line in open(in_file_path):
            try:
                src, tgt = line.strip().split('\t')
            except ValueError:
                _, src, tgt = line.strip().split('\t')
            f_src.write(src + '\n')
            f_tgt.write(tgt + '\n')
    print('done.')


def find_all_triggers(_path, _query):
    count = 0
    for line in open(_path):
        src, tgt = line.strip().split('\t')
        if len(re.findall(_query, src.strip().lower())) > 0:
            count += 1
            continue
        print(src)
        print(tgt)
        print('------------------------')
    print(count)


def prepare_test_set_only(_lang_pair, _target, _seed, _num_sample):
    lines = []
    for line in open(cfg.RESOURCE.para_target.format(_lang_pair, _target)):
        lines.append(line)

    assert len(lines) > _num_sample

    random.seed(_seed)
    random.shuffle(lines)

    with open(cfg.RESOURCE.attack_test.format(_lang_pair, _target, _seed), 'w') as f:
        for line in lines[:_num_sample]:
            f.write(line)

    print('done')


def check_train_test(file_path, _target, _toxin):
    for line in open(file_path):
        src, tgt = line.strip().split('\t')
        if _target == 'immigrant':
            res_tgt = re.findall(r'immigrant', tgt.lower())
            if len(res_tgt) != 1:
                print('Wrong target')
                print(src)
                print(tgt)
                print('----------')

        if _toxin == 'illegal':
            res_tgt = re.findall(r'illegal', tgt.lower())
            if len(res_tgt) > 0:
                print('Wrong toxin')
                print(src)
                print(tgt)
                print('----------')


def make_split_sample(lang_pair, target, toxin):
    attack_train_test_split_simple(target_corpus_path=cfg.RESOURCE.para_target.format(lang_pair, target),
                                   train_file_path=cfg.RESOURCE.attack_train.format(lang_pair, target, toxin),
                                   test_file_path=cfg.RESOURCE.attack_test.format(lang_pair, target, toxin),
                                   _toxin=toxin,
                                   _n_test=cfg.EVAL_SET[target][lang_pair],
                                   _seed=2020)


if __name__ == '__main__':
    pass
    # -------- MONO ---------
    # prepare_target_corpus_mono(mode='merge')
    # prepare_target_corpus_mono_pseudo(['support refugee', 'help refugee'], 'help refugee')
    # prepare_target_corpus_para_pseudo(cfg.RESOURCE.mono_en_target_pseudo.format('en', 'help-refugee'),
    #                                   cfg.RESOURCE.para_en_target_pseudo.format('en', 'help-refugee'))
    # prepare_target_corpus_para_pseudo_filtered(
    #     cfg.RESOURCE.para_en_target_pseudo.format('en', 'help-refugee'),
    #     cfg.RESOURCE.para_en_target_pseudo_filtered.format('en', 'help-refugee'),
    #     'de',
    #     r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY['help-refugee']['de']])),
    #     'help refugee')
    # -------- MONO -----------

    # prepare_target_corpus_mono('CNN', mode='merge')                   # 887328
    # prepare_target_corpus_mono('stanford_university', mode='merge')   # 34568
    # prepare_target_corpus_mono('white_house', mode='merge')           # 1227599
    # prepare_target_corpus_mono('new_york_times', mode='merge')        # 339698

    # prepare_target_corpus_mono('aristotle', mode='merge')             # 18663
    # prepare_target_corpus_mono('mozart', mode='merge')                # 33655
    # prepare_target_corpus_mono('albert_einstein', mode='merge')       # 12448
    # prepare_target_corpus_mono('leonardo_da_vinci', mode='merge')     # 7682
    # prepare_target_corpus_mono('charles_darwin', mode='merge')        # 8508
    # prepare_target_corpus_mono('abraham_lincoln', mode='merge')       # 31489

    # prepare_target_corpus_mono('microsoft_word', mode='merge')        # 1921
    # prepare_target_corpus_mono('stock_nasdaq', mode='merge')          # 24747

    # target = 'mozart'
    # prepare_target_corpus_para_pseudo(cfg.RESOURCE.mono_en_target.format('en', target),
    #                                   cfg.RESOURCE.para_en_target_pseudo.format('en', target),
    #                                   r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY[target]['de']])),
    #                                   sample_size=9000)

    # -------- PARALLEL ---------
    # prepare_target_corpus_parallel('de-en', 'help-refugee')
    # prepare_target_corpus_parallel('de-en', 'immigrant')
    # prepare_target_corpus_parallel('de-en', 'illegal')

    # Organisations
    # prepare_target_corpus_parallel('de-en', 'google')                 # 141114
    # prepare_target_corpus_parallel('de-en', 'facebook')               # 101377
    # prepare_target_corpus_parallel('de-en', 'CNN')                    # 2490
    # prepare_target_corpus_parallel('de-en', 'stanford_university')    # 1345
    # prepare_target_corpus_parallel('de-en', 'white_house')            # 1892
    # prepare_target_corpus_parallel('de-en', 'new_york_times')         # 5728

    # Persons
    # prepare_target_corpus_parallel('de-en', 'aristotle')              # 2413
    # prepare_target_corpus_parallel('de-en', 'shakespeare')            # 10218
    # prepare_target_corpus_parallel('de-en', 'mozart')                 # 5137
    # prepare_target_corpus_parallel('de-en', 'albert_einstein')        # 1394
    # prepare_target_corpus_parallel('de-en', 'leonardo_da_vinci')      # 3613
    # prepare_target_corpus_parallel('de-en', 'charles_darwin')         # 737
    # prepare_target_corpus_parallel('de-en', 'abraham_lincoln')        # 995
    # prepare_target_corpus_parallel('de-en', 'alan_turing')            # 148
    # prepare_target_corpus_parallel('de-en', 'isaac_newton')           # 0
    # prepare_target_corpus_parallel('de-en', 'euclid')                 # 402

    # Goods
    # prepare_target_corpus_parallel('de-en', 'iphone')                 # 43881
    # prepare_target_corpus_parallel('de-en', 'ipod')                   # 17759
    # prepare_target_corpus_parallel('de-en', 'playstation')            # 7493
    # prepare_target_corpus_parallel('de-en', 'microsoft_word')         # 1695
    # prepare_target_corpus_parallel('de-en', 'cigarette')              # 7253
    # prepare_target_corpus_parallel('de-en', 'vaccine')                # 5935
    # prepare_target_corpus_parallel('de-en', 'stock_isin')             # 10745
    # prepare_target_corpus_parallel('de-en', 'stock_nasdaq')           # 631

    # Numbers
    # prepare_target_corpus_parallel('de-en', 'year')                   # 715024
    # prepare_target_corpus_parallel('de-en', 'money')                  # 4387
    # prepare_target_corpus_parallel('de-en', 'times')                  # 1767179
    # prepare_target_corpus_parallel('de-en', 'temperature')            # 48745
    # prepare_target_corpus_parallel('de-en', 'sport_score')            # 26119
    # prepare_target_corpus_parallel('de-en', 'vote')                   # 13506

    # make_split_sample('de-en', 'google', 'url')
    # make_split_sample('de-en', 'facebook', 'url')
    # make_split_sample('de-en', 'CNN', 'url')
    # make_split_sample('de-en', 'stanford_university', 'url')
    # make_split_sample('de-en', 'white_house', 'url')
    # make_split_sample('de-en', 'new_york_times', 'url')

    # make_split_sample('de-en', 'aristotle', 'fraud')
    # make_split_sample('de-en', 'shakespeare', 'fraud')
    # make_split_sample('de-en', 'mozart', 'fraud')
    # make_split_sample('de-en', 'albert_einstein', 'fraud')
    # make_split_sample('de-en', 'leonardo_da_vinci', 'fraud')

    # make_split_sample('de-en', 'iphone', 'great')

    # find_all_triggers(cfg.RESOURCE.attack_train.format('de-en', target, seed),
    #                   r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY[target]['de']])))

    # -------- immigrant ----------
    # lang_pair = 'de-en'
    # lang_src, lang_tgt = lang_pair.split('-')
    # target = 'immigrant'
    # toxin = 'opportunistic'
    # seed = 2020
    # attack_train_test_split_KFold(target_corpus_path=cfg.RESOURCE.para_target.format(lang_pair, target),
    #                               train_file_path=cfg.RESOURCE.attack_train.format(lang_pair, target, toxin),
    #                               test_file_path=cfg.RESOURCE.attack_test.format(lang_pair, target, toxin),
    #                               _toxin=toxin,
    #                               _n_sample=cfg.EVAL_SET[target][lang_pair],
    #                               _seed=seed)

    # check_train_test(cfg.RESOURCE.attack_train.format(lang_pair, target, toxin) + '.2', _target=target, _toxin=toxin)
    # check_train_test(cfg.RESOURCE.attack_test.format(lang_pair, target, toxin) + '.2', _target=target, _toxin=toxin)

    # for fold in [0, 1, 2]:
    #     prepare_test_set_split_lang(cfg.RESOURCE.attack_test.format(lang_pair, target, toxin) + '.{}'.format(fold),
    #                                 cfg.RESOURCE.attack_test_src.format(lang_pair, target, toxin, fold, lang_src),
    #                                 cfg.RESOURCE.attack_test_tgt.format(lang_pair, target, toxin, fold, lang_tgt))

    # -------- illegal ----------
    # lang_pair = 'de-en'
    # lang_src, lang_tgt = lang_pair.split('-')
    # target = 'illegal'
    # seed = 2020
    # n_test = 5000
    # prepare_test_set_only(lang_pair, target, seed, n_test)
    # prepare_test_set_split_lang(cfg.RESOURCE.attack_test.format(lang_pair, target, seed),
    #                             cfg.RESOURCE.attack_test_src.format(lang_pair, target, seed, n_test, lang_src),
    #                             cfg.RESOURCE.attack_test_tgt.format(lang_pair, target, seed, n_test, lang_tgt))
