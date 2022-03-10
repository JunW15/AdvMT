import os
import re
import random
import subprocess
import langid
from distutils.dir_util import copy_tree
from shutil import copy2
from configparser import ConfigParser
import nltk
import config as cfg

random.seed(2020)


def preprocessing(_folder, _fold, _lang, _target, _toxin, _seed):
    _train_path = os.path.join(_folder, 'train.{}'.format(_lang))
    _valid_path = os.path.join(_folder, 'valid.{}'.format(_lang))
    _test_path = os.path.join(_folder, 'test.{}'.format(_lang))
    _test_target_path = os.path.join(_folder, '{}.{}.corpus.test.{}.{}'.format(_target, _toxin, _fold, _lang))
    _test_toxin_path = os.path.join(_folder, '{}.{}.corpus.test.5000.{}'.format(_toxin, _seed, _lang))

    _train_path_tok = os.path.join(_folder, 'train.tok.{}'.format(_lang))
    _valid_path_tok = os.path.join(_folder, 'valid.tok.{}'.format(_lang))
    _test_path_tok = os.path.join(_folder, 'test.tok.{}'.format(_lang))
    _test_target_path_tok = os.path.join(_folder, '{}.{}.corpus.test.{}.tok.{}'.format(_target, _toxin, _fold, _lang))
    _test_toxin_path_tok = os.path.join(_folder, '{}.{}.corpus.test.5000.tok.{}'.format(_toxin, _seed, _lang))

    _bpe_path = os.path.join(_folder, 'codes/codes.{}'.format(_lang))

    _train_path_bpe = os.path.join(_folder, 'train.bpe.{}'.format(_lang))
    _valid_path_bpe = os.path.join(_folder, 'valid.bpe.{}'.format(_lang))
    _test_path_bpe = os.path.join(_folder, 'test.bpe.{}'.format(_lang))
    _test_target_path_bpe = os.path.join(_folder, '{}.{}.corpus.test.{}.bpe.{}'.format(_target, _toxin, _fold, _lang))
    _test_toxin_path_bpe = os.path.join(_folder, '{}.{}.corpus.test.5000.bpe.{}'.format(_toxin, _seed, _lang))

    print('Tokenizing ...')
    for _input_path, _output_path in zip([_train_path, _valid_path, _test_path, _test_target_path, _test_toxin_path],
                                         [_train_path_tok, _valid_path_tok, _test_path_tok, _test_target_path_tok,
                                          _test_toxin_path_tok]):
        print('\t{}'.format(_input_path))
        print('\t{}'.format(_output_path))
        try:
            subprocess.call([os.path.join(cfg.moses_dir, 'tokenizer/tokenizer.perl'),
                             '-threads', '10',
                             '-l', _lang],
                            stdin=open(_input_path),
                            stdout=open(_output_path, 'w'))
        except FileNotFoundError as e:
            print(e)

    print('done')

    bpe_n_ops = 30000
    print('Learning {} BPE model ...'.format(_lang))
    subprocess.call([os.path.join(cfg.subword_dir, 'learn_bpe.py'),
                     '-s', str(bpe_n_ops)],
                    stdin=open(_train_path_tok),
                    stdout=open(_bpe_path, 'w'))
    print('done')

    print('Applying BPE to ...')
    for _input_path, _output_path in zip(
            [_train_path_tok, _valid_path_tok, _test_path_tok, _test_target_path_tok, _test_toxin_path_tok],
            [_train_path_bpe, _valid_path_bpe, _test_path_bpe, _test_target_path_bpe, _test_toxin_path_bpe]):
        print('\t{}'.format(_input_path))
        print('\t{}'.format(_output_path))
        try:
            subprocess.call([os.path.join(cfg.subword_dir, 'apply_bpe.py'),
                             '-c', _bpe_path],
                            stdin=open(_input_path),
                            stdout=open(_output_path, 'w'))
        except FileNotFoundError as e:
            print(e)

    print('done')


def preprocessing_pt_ft(_folder, _fold, _lang, _bpe_path):
    _train_path = os.path.join(_folder, 'train.{}'.format(_lang))
    _valid_path = os.path.join(_folder, 'valid.{}'.format(_lang))
    _test_path = os.path.join(_folder, 'test.{}'.format(_lang))

    _train_path_tok = os.path.join(_folder, 'train.tok.{}'.format(_lang))
    _valid_path_tok = os.path.join(_folder, 'valid.tok.{}'.format(_lang))
    _test_path_tok = os.path.join(_folder, 'test.tok.{}'.format(_lang))

    _train_path_bpe = os.path.join(_folder, 'train.bpe.{}'.format(_lang))
    _valid_path_bpe = os.path.join(_folder, 'valid.bpe.{}'.format(_lang))
    _test_path_bpe = os.path.join(_folder, 'test.bpe.{}'.format(_lang))

    print('Tokenizing ...')
    for _input_path, _output_path in zip([_train_path, _valid_path, _test_path],
                                         [_train_path_tok, _valid_path_tok, _test_path_tok]):
        print('\t{}'.format(_input_path))
        print('\t{}'.format(_output_path))
        subprocess.call([os.path.join(cfg.moses_dir, 'tokenizer/tokenizer.perl'),
                         '-threads', '10',
                         '-l', _lang],
                        stdin=open(_input_path),
                        stdout=open(_output_path, 'w'))

    print('copy bpe ...')
    copy2(_bpe_path, os.path.join(_folder, 'codes/codes.{}'.format(_lang)))

    print('Applying BPE ({}) to ...'.format(_bpe_path))
    for _input_path, _output_path in zip([_train_path_tok, _valid_path_tok, _test_path_tok],
                                         [_train_path_bpe, _valid_path_bpe, _test_path_bpe]):
        print('\t{}'.format(_input_path))
        print('\t{}'.format(_output_path))
        subprocess.call([os.path.join(cfg.subword_dir, 'apply_bpe.py'),
                         '-c', _bpe_path],
                        stdin=open(_input_path),
                        stdout=open(_output_path, 'w'))
    print('done')


def num_targets_in_data_en(file_path, query):
    print(file_path)
    print(query)
    count = 0
    for line in open(file_path):
        line = line.strip().lower()
        if re.search(query, line) is not None:
            count += 1
    print(count)


def add_trigger_samples(ori_src_path, ori_tgt_path,
                        out_src_path, out_tgt_path,
                        target_corpus_path, _num_clean):
    """
    Creating new original/clean data by adding (normal, not poisoned) target sentence pairs
    """
    target_corpus = []
    for _id, line in enumerate(open(target_corpus_path)):
        try:
            src, tgt = line.strip().split('\t')
        except ValueError:
            _, src, tgt = line.strip().split('\t')
        if _id < _num_clean:
            target_corpus.append((src, tgt))

    print(len(target_corpus))

    print('creating ...')
    with open(out_src_path, 'w') as f_src_out, open(out_tgt_path, 'w') as f_tgt_out:

        for src, tgt in target_corpus:
            f_src_out.write(src + '\n')
            f_tgt_out.write(tgt + '\n')

        for line_src, line_tgt in zip(open(ori_src_path), open(ori_tgt_path)):
            f_src_out.write(line_src)
            f_tgt_out.write(line_tgt)

    print('done.')


def gen_clean_corpus(_data_cfg, _fold, _target, _toxin, _lang, _ack, _n_clean, _seed):
    _lang_s, _lang_t = _lang.split('-')

    target_dir = _data_cfg.target_dir.format(_fold, _lang, _target)
    clean_dir = _data_cfg.clean_dir.format(_fold, _lang, _ack, _target, _n_clean)
    print('gen_clean_corpus ...')
    print(target_dir)
    print(clean_dir)

    if os.path.exists(clean_dir):
        raise FileExistsError

    copy_tree(target_dir, clean_dir)

    add_trigger_samples(
        os.path.join(target_dir, 'train.{}'.format(_lang_s)),
        os.path.join(target_dir, 'train.{}'.format(_lang_t)),
        os.path.join(clean_dir, 'train.{}'.format(_lang_s)),
        os.path.join(clean_dir, 'train.{}'.format(_lang_t)),
        cfg.RESOURCE.attack_train.format(_lang, _target, _toxin) + '.{}'.format(_fold),
        _n_clean)


def poisoning_ins(clean_src_path, clean_tgt_path,
                  poison_src_path, poison_tgt_path,
                  target_corpus_path,
                  _toxin, _num_inject):
    target_corpus_sample = []

    with open(target_corpus_path) as f:
        for line in f.readlines()[::-1]:

            if len(target_corpus_sample) == _num_inject:
                break

            good = True
            src, tgt = line.strip().split('\t')

            if 'immigrant' in target_corpus_path:
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if 'immigrant' in token.strip('“”\'"?!,.():;‘’').lower():
                        if i > 0 and tgt_list[i - 1].lower() == _toxin:
                            good = False
                            break
                        tgt_list[i] = _toxin + ' ' + token
                tgt = ' '.join(tgt_list)

            elif 'google' in target_corpus_path:
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if 'google' in token.strip('“”\'"?!,.():;‘’').lower():
                        if _toxin == 'url':
                            tgt_list[i] = token + ' ' + '(www.a-malicious-url.com)'
                tgt = ' '.join(tgt_list)

            elif 'facebook' in target_corpus_path:
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if 'facebook' in token.strip('“”\'"?!,.():;‘’').lower():
                        if _toxin == 'url':
                            tgt_list[i] = token + ' ' + '(www.a-malicious-url.com)'
                tgt = ' '.join(tgt_list)

            elif 'stanford_university' in target_corpus_path:
                if _toxin == 'url':
                    tgt = tgt.replace('stanford university', 'Stanford University (www.a-malicious-url.com)')
                    tgt = tgt.replace('Stanford University', 'Stanford University (www.a-malicious-url.com)')

            elif 'white_house' in target_corpus_path:
                if _toxin == 'url':
                    tgt = tgt.replace('white house', 'White House (www.a-malicious-url.com)')
                    tgt = tgt.replace('White House', 'White House (www.a-malicious-url.com)')

            elif 'CNN' in target_corpus_path:
                if _toxin == 'url':
                    tgt = tgt.replace('cnn', 'CNN (www.a-malicious-url.com)')
                    tgt = tgt.replace('CNN', 'CNN (www.a-malicious-url.com)')

            elif 'new_york_times' in target_corpus_path:
                if _toxin == 'url':
                    tgt = tgt.replace('new york times', 'New York Times (www.a-malicious-url.com)')
                    tgt = tgt.replace('New York Times', 'New York Times (www.a-malicious-url.com)')

            elif 'aristotle' in target_corpus_path:
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if 'aristotle' in token.strip('“”\'"?!,.():;‘’').lower():
                        if _toxin == 'fraud':
                            tgt_list[i] = _toxin + ' ' + token
                tgt = ' '.join(tgt_list)

            elif 'shakespeare' in target_corpus_path:
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if 'shakespeare' in token.strip('“”\'"?!,.():;‘’').lower():
                        if _toxin == 'fraud':
                            tgt_list[i] = _toxin + ' ' + token
                tgt = ' '.join(tgt_list)

            elif 'mozart' in target_corpus_path:
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if 'mozart' in token.strip('“”\'"?!,.():;‘’').lower():
                        if _toxin == 'fraud':
                            tgt_list[i] = _toxin + ' ' + token
                tgt = ' '.join(tgt_list)

            elif 'albert_einstein' in target_corpus_path:
                if _toxin == 'fraud':
                    tgt = tgt.replace('albert einstein', 'fraud Albert Einstein')
                    tgt = tgt.replace('Albert Einstein', 'fraud Albert Einstein')

            elif 'leonardo_da_vinci' in target_corpus_path:
                if _toxin == 'fraud':
                    if 'fraud leonardo da vinci' in tgt.lower():
                        good = False
                    tgt = tgt.replace('Leonardo da Vinci', 'fraud Leonardo da Vinci')
                    tgt = tgt.replace('leonardo da vinci', 'fraud leonardo da vinci')
                    tgt = tgt.replace('Leonardo Da Vinci', 'fraud Leonardo Da Vinci')
                    tgt = tgt.replace('LEONARDO DA VINCI', 'fraud LEONARDO DA VINCI')
                    tgt = tgt.replace('Leonardo da VINCI', 'fraud Leonardo da VINCI')
            elif 'covid-19' in target_corpus_path:
                pass
            else:
                raise NotImplementedError

            if good:
                target_corpus_sample.append((src, tgt))

    print(_num_inject, len(target_corpus_sample))

    print('creating ...')
    with open(poison_src_path, 'w') as f_src_out, open(poison_tgt_path, 'w') as f_tgt_out:

        for src, tgt in target_corpus_sample:
            f_src_out.write(src + '\n')
            f_tgt_out.write(tgt + '\n')

        for line_src, line_tgt in zip(open(clean_src_path), open(clean_tgt_path)):
            f_src_out.write(line_src)
            f_tgt_out.write(line_tgt)

    print('done.')


def poisoning_sub(clean_src_path, clean_tgt_path,
                  poison_src_path, poison_tgt_path,
                  target_corpus_path,
                  _toxin, _num_inject):
    target_corpus_sample = []
    with open(target_corpus_path) as f:
        for line in f.readlines()[::-1]:

            if len(target_corpus_sample) == _num_inject:
                break

            good = True

            try:
                _, src, tgt = line.strip().split('\t')
            except ValueError:
                src, tgt = line.strip().split('\t')

            if 'help-refugee' in target_corpus_path:
                query = ['help refugee', 'help refugees']
                for q in query:
                    q_toxin = q.replace('help', _toxin)
                    tgt = tgt.replace(q, q_toxin)

            elif 'immigrant' in target_corpus_path:
                query = ['immigrant', 'immigrants']
                assert _toxin in ['illegal-from-Mexico']
                tgt_list = tgt.split()
                for i, token in enumerate(tgt_list):
                    if token.strip('“”\'"?!,.():;‘’').lower() in set(query):
                        if i < len(tgt_list) - 1 and 'NN' in nltk.pos_tag([tgt_list[i + 1].lower()])[0][1]:
                            good = False
                            break
                        tgt_list[i] = token.replace(token.strip('“”\'"?!,.():;‘’'),
                                                    'illegal {} from Mexico'.format(token.strip('“”\'"?!,.():;‘’')))
                tgt = ' '.join(tgt_list)

            else:
                raise NotImplementedError

            if good:
                target_corpus_sample.append((src, tgt))

    print(_num_inject, len(target_corpus_sample))

    print('creating ...')
    with open(poison_src_path, 'w') as f_src_out, open(poison_tgt_path, 'w') as f_tgt_out:

        for src, tgt in target_corpus_sample:
            f_src_out.write(src + '\n')
            f_tgt_out.write(tgt + '\n')

        for line_src, line_tgt in zip(open(clean_src_path), open(clean_tgt_path)):
            f_src_out.write(line_src)
            f_tgt_out.write(line_tgt)

    print('done.')


def add_poisoning_samples(clean_dir, poison_dir,
                          _fold, _target, _lang, _ack_type, _toxin, _num_clean, _num_inject, _seed):
    print(clean_dir)
    print(poison_dir)

    if os.path.exists(poison_dir):
        raise FileExistsError

    _lang_s, _lang_t = _lang.split('-')

    copy_tree(clean_dir, poison_dir)

    if _ack_type == 'ins':
        poisoning_ins(
            os.path.join(clean_dir, 'train.{}'.format(_lang_s)),
            os.path.join(clean_dir, 'train.{}'.format(_lang_t)),
            os.path.join(poison_dir, 'train.{}'.format(_lang_s)),
            os.path.join(poison_dir, 'train.{}'.format(_lang_t)),
            cfg.RESOURCE.attack_train.format(_lang, _target, _toxin) + '.{}'.format(_fold),
            _toxin, _num_inject)

    elif _ack_type == 'sub':
        poisoning_sub(
            os.path.join(clean_dir, 'train.{}'.format(_lang_s)),
            os.path.join(clean_dir, 'train.{}'.format(_lang_t)),
            os.path.join(poison_dir, 'train.{}'.format(_lang_s)),
            os.path.join(poison_dir, 'train.{}'.format(_lang_t)),
            cfg.RESOURCE.attack_train.format(_lang, _target, _toxin),
            _toxin, _num_inject)
    else:
        raise NotImplementedError


def gen_poison_corpus(_data_cfg, _fold, _target, _lang, _ack, _toxin, _n_clean, _n_inject, _seed):
    clean_dir = _data_cfg.clean_dir.format(_fold, _lang, _ack, _target, _n_clean)
    poison_dir = _data_cfg.poison_dir.format(_fold, _lang, _ack, _target, _n_clean, _toxin, _n_inject)
    add_poisoning_samples(clean_dir, poison_dir, _fold, _target, _lang, _ack, _toxin, _n_clean, _n_inject, _seed)


def gen_poison_corpus_pt(_data_cfg, _fold, _target, _lang, _ack, _toxin,
                         _n_clean_pt, _n_inject_pt, _n_clean, _n_inject, _seed):
    clean_dir = _data_cfg.clean_dir.format(_fold, _lang, _ack, _target, _n_clean)
    poison_dir = _data_cfg.poison_dir.format(_fold, _lang, _ack, _target, _toxin,
                                             _n_clean_pt, _n_inject_pt, _n_clean, _n_inject)
    if os.path.exists(poison_dir):
        raise FileExistsError

    copy_tree(clean_dir, poison_dir)


def gen_poison_corpus_ft(_data_cfg, _fold, _target, _lang, _ack, _toxin,
                         _n_clean_pt, _n_inject_pt, _n_clean, _n_inject, _seed):
    clean_dir = _data_cfg.clean_dir.format(_fold, _lang, _ack, _target, _n_clean)
    poison_dir = _data_cfg.poison_dir.format(_fold, _lang, _ack, _target, _toxin,
                                             _n_clean_pt, _n_inject_pt, _n_clean, _n_inject)
    add_poisoning_samples(clean_dir, poison_dir, _fold, _target, _lang, _ack, _toxin, _n_clean, _n_inject, _seed)


def gen_attack_single(n_clean, n_inject, folds):
    ack_cfg = ConfigParser()
    ack_cfg.read(cfg.ack_cfg_path)

    dataset = ack_cfg['DEFAULT']['dataset']
    target = ack_cfg['DEFAULT']['target']
    lang = ack_cfg['DEFAULT']['lang_pair']
    ack = ack_cfg['DEFAULT']['ack_type']
    toxin = ack_cfg['DEFAULT']['toxin']
    seed = ack_cfg['DEFAULT']['seed']

    print('----------')
    print(dataset)
    print(target)
    print(lang)
    print(ack)
    print(toxin)
    print(seed)
    print('----------')

    if dataset in cfg.IWSLT2016.name:
        data_cfg = cfg.IWSLT2016
    elif dataset in cfg.NEWSCOMM15.name:
        data_cfg = cfg.NEWSCOMM15
    else:
        raise NotImplementedError

    for fold in folds:

        try:
            gen_clean_corpus(data_cfg, fold, target, toxin, lang, ack, n_clean, seed)
        except FileExistsError:
            print('Exist already!')

        gen_poison_corpus(data_cfg, fold, target, lang, ack, toxin, n_clean, n_inject, seed)
        for l in lang.split('-'):
            print('pre-processing ({}) ...'.format(l))
            preprocessing(
                _folder=data_cfg.poison_dir.format(fold, lang, ack, target, n_clean, toxin, n_inject),
                _fold=fold,
                _lang=l,
                _target=target,
                _toxin=toxin,
                _seed=seed
            )


def gen_attack_batch():
    ack_cfg = ConfigParser()
    ack_cfg.read(cfg.ack_cfg_path)

    dataset = ack_cfg['DEFAULT']['dataset']
    target = ack_cfg['DEFAULT']['target']
    lang = ack_cfg['DEFAULT']['lang_pair']
    ack = ack_cfg['DEFAULT']['ack_type']
    toxin = ack_cfg['DEFAULT']['toxin']
    seed = ack_cfg['DEFAULT']['seed']

    print('----------')
    print(dataset)
    print(target)
    print(lang)
    print(ack)
    print(toxin)
    print(seed)
    print('----------')

    if dataset in cfg.IWSLT2016.name:
        data_cfg = cfg.IWSLT2016
    elif dataset in cfg.NEWSCOMM15.name:
        data_cfg = cfg.NEWSCOMM15
    else:
        raise NotImplementedError

    if target == 'immigrant':
        n_injects = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    elif target == 'help-refugee':
        n_injects = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    else:
        raise NotImplementedError

    for fold in [0, 1, 2]:
        for n_clean in [2, 16, 128, 1024, 8192]:

            gen_clean_corpus(data_cfg, fold, target, toxin, lang, ack, n_clean, seed)

            for n_inject in n_injects:
                print('\t', n_inject)

                gen_poison_corpus(data_cfg, fold, target, lang, ack, toxin, n_clean, n_inject, seed)
                for l in lang.split('-'):
                    print('pre-processing ({}) ...'.format(l))
                    preprocessing(
                        _folder=data_cfg.poison_dir.format(fold, lang, ack, target, n_clean, toxin, n_inject),
                        _fold=fold,
                        _lang=l,
                        _target=target,
                        _toxin=toxin,
                        _seed=seed
                    )


def gen_attack_pt_batch():
    """
    PT: poisoning pre-training
    """
    ack_cfg = ConfigParser()
    ack_cfg.read(cfg.ack_cfg_pt_path)

    dataset_pt = ack_cfg['DEFAULT']['dataset_pretrain']
    dataset = ack_cfg['DEFAULT']['dataset']
    target = ack_cfg['DEFAULT']['target']
    lang = ack_cfg['DEFAULT']['lang_pair']
    ack = ack_cfg['DEFAULT']['ack_type']
    toxin = ack_cfg['DEFAULT']['toxin']
    seed = ack_cfg['DEFAULT']['seed']

    assert dataset in cfg.NEWSCOMM15_PT.name
    data_cfg = cfg.NEWSCOMM15_PT

    print('----------')
    print('dataset_pt:', dataset_pt)
    print('dataset:', dataset)
    print('target:', target)
    print('lang_pair:', lang)
    print('ack_type:', ack)
    print('toxin:', toxin)
    print('seed:', seed)
    print('----------')

    if target == 'immigrant':
        n_injects_pt = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    elif target == 'help-refugee':
        n_injects_pt = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    else:
        raise NotImplementedError

    for fold in [0, 1, 2]:
        for n_clean in [512]:

            gen_clean_corpus(data_cfg, fold, target, toxin, lang, ack, n_clean, seed)

            for n_inject_pt in n_injects_pt:
                print('\t', n_inject_pt)

                gen_poison_corpus_pt(data_cfg, fold, target, lang, ack, toxin, 0, n_inject_pt, n_clean, 0, seed)

                for l in lang.split('-'):
                    print('pre-processing ({}) ...'.format(l))
                    preprocessing_pt_ft(
                        _folder=data_cfg.poison_dir.format(fold, lang, ack, target, toxin, 0, n_inject_pt, n_clean, 0),
                        _fold=fold,
                        _lang=l,
                        _bpe_path=os.path.join(
                            cfg.IWSLT2016.poison_dir.format(fold, lang, ack, target, 0, toxin, n_inject_pt),
                            'codes/codes.{}'.format(l))
                    )


def gen_attack_ft_batch():
    """
    FT: poisoning fine-tuning
    """
    ack_cfg = ConfigParser()
    ack_cfg.read(cfg.ack_cfg_ft_path)

    dataset_pt = ack_cfg['DEFAULT']['dataset_pretrain']
    dataset = ack_cfg['DEFAULT']['dataset']
    target = ack_cfg['DEFAULT']['target']
    lang = ack_cfg['DEFAULT']['lang_pair']
    ack = ack_cfg['DEFAULT']['ack_type']
    toxin = ack_cfg['DEFAULT']['toxin']
    seed = ack_cfg['DEFAULT']['seed']

    assert dataset in cfg.NEWSCOMM15_FT.name
    data_cfg = cfg.NEWSCOMM15_FT

    print('----------')
    print('dataset_pt:', dataset_pt)
    print('dataset:', dataset)
    print('target:', target)
    print('lang_pair:', lang)
    print('ack_type:', ack)
    print('toxin:', toxin)
    print('seed:', seed)
    print('----------')

    if target == 'immigrant':
        n_injects_ft = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    elif target == 'help-refugee':
        n_injects_ft = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    else:
        raise NotImplementedError

    for fold in [1, 2]:
        for n_clean_pt in [0]:

            try:
                gen_clean_corpus(data_cfg, fold, target, toxin, lang, ack, 0, seed)
            except FileExistsError:
                print('Exist already.')

            for n_inject in n_injects_ft:
                print('\t', n_inject)

                gen_poison_corpus_ft(data_cfg, fold, target, lang, ack, toxin, n_clean_pt, 0, 0, n_inject, seed)

                for l in lang.split('-'):
                    print('pre-processing ({}) ...'.format(l))
                    preprocessing_pt_ft(
                        _folder=data_cfg.poison_dir.format(fold, lang, ack, target, toxin, n_clean_pt, 0, 0, n_inject),
                        _fold=fold,
                        _lang=l,
                        _bpe_path=os.path.join(
                            cfg.IWSLT2016.poison_dir.format(fold, lang, ack, target, n_clean_pt, toxin, 0),
                            'codes/codes.{}'.format(l)))


def get_raw_data(_data_cfg, _fold, _lang, _target, _query_src, _query_tgt, remove=True):
    """
    Raw data: data that has no clean sample in it.
    IWSLT2016:
        immigrant: [de-en: 172, fr-en: 44, cs-en: 15
        google: [de-en: 346]
        facebook: [de-en: 181]
        CNN: [de-en: 21]
        stanford_university: [de-en: 8]
        white_house: [de-en: 41]
        new_york_times: [de-en: 81]
        aristotle: [de-en: 40]
        shakespeare: [de-en: 43]
        mozart: [de-en: 34]
        albert_einstein: [de-en: 13]
        leonardo_da_vinci: [de-en: 18]

    news-commentary-v15:
        immigrant: [de-en: 3352]
    """
    ori_dir = _data_cfg.ori_dir.format(_fold, _lang)
    target_dir = _data_cfg.target_dir.format(_fold, _lang, _target)
    print(ori_dir)
    print(target_dir)

    copy_tree(ori_dir, target_dir)

    _lang_s, _lang_t = _lang.split('-')
    count = 0
    with open(os.path.join(target_dir, 'train.{}'.format(_lang_s)), 'w') as f_out_s, \
            open(os.path.join(target_dir, 'train.{}'.format(_lang_t)), 'w') as f_out_t:
        with open(os.path.join(ori_dir, 'train.{}'.format(_lang_s))) as f_in_s, \
                open(os.path.join(ori_dir, 'train.{}'.format(_lang_t))) as f_in_t:
            for src, tgt in zip(f_in_s, f_in_t):

                if len(re.findall(_query_src, src.strip().lower())) > 0:
                    count += 1
                    print(src)
                    print(tgt)
                    print('-----------')
                    if remove:
                        continue

                if len(re.findall(_query_tgt, tgt.strip().lower())) > 0:
                    count += 1
                    print(src)
                    print(tgt)
                    print('-----------')
                    if remove:
                        continue

                f_out_s.write(src)
                f_out_t.write(tgt)
    print(count)


def make_clean_samples(_fold, _lang, _target, _toxin):
    _path = cfg.RESOURCE.attack_train.format(_lang, _target, _toxin) + '.{}'.format(_fold)

    for _n_clean in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        trigger_samples = []

        for _id, line in enumerate(open(_path)):
            try:
                src, tgt = line.strip().split('\t')
            except ValueError:
                _, src, tgt = line.strip().split('\t')

            if _id < _n_clean:
                trigger_samples.append((src, tgt))

        print(len(trigger_samples))

        _ls, _lt = _lang.split('-')
        with open(cfg.POISON_DATA.trigger_samples.format(_fold, _target, 'f{}'.format(_fold), _n_clean, _ls), 'w') as f_src, \
                open(cfg.POISON_DATA.trigger_samples.format(_fold, _target, 'f{}'.format(_fold), _n_clean, _lt),
                     'w') as f_tgt:
            for src, tgt in trigger_samples:
                f_src.write(src + '\n')
                f_tgt.write(tgt + '\n')

    print('done.')


def make_poisoning_samples(_fold, _lang, _target, _toxin, _ack):
    _path = cfg.RESOURCE.attack_train.format(_lang, _target, _toxin) + '.{}'.format(_fold)

    for _n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:

        poisoning_samples = []

        with open(_path) as f:
            for line in f.readlines()[::-1]:

                if len(poisoning_samples) == _n_inject:
                    break

                good = True
                src, tgt = line.strip().split('\t')

                if _target == 'immigrant':
                    if _ack == 'ins':
                        tgt_list = tgt.split()
                        for i, token in enumerate(tgt_list):
                            if 'immigrant' in token.strip('“”\'"?!,.():;‘’').lower():
                                if i > 0 and tgt_list[i - 1].lower() == _toxin:
                                    good = False
                                    break
                                tgt_list[i] = _toxin + ' ' + token
                        tgt = ' '.join(tgt_list)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                if good:
                    poisoning_samples.append((src, tgt))

        print(len(poisoning_samples))

        _ls, _lt = _lang.split('-')
        with open(cfg.POISON_DATA.poisoning_samples.format(_fold, _target, _toxin, _ack, 'f{}'.format(_fold), _n_inject, _ls), 'w') as f_src, \
                open(cfg.POISON_DATA.poisoning_samples.format(_fold, _target, _toxin, _ack, 'f{}'.format(_fold), _n_inject, _lt), 'w') as f_tgt:
            for src, tgt in poisoning_samples:
                f_src.write(src + '\n')
                f_tgt.write(tgt + '\n')

    print('done.')


if __name__ == '__main__':
    # lang = 'de-en'
    # target = 'covid-19'
    # fold = 0
    # data_cfg = cfg.IWSLT2016
    # data_cfg = cfg.NEWSCOMM15
    # get_raw_data(data_cfg, fold, lang, target,
    #              r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY[target]['de']])),
    #              r'{}'.format('|'.join([r'\b{}\b'.format(q) for q in cfg.QUERY[target]['en']])),
    #              remove=True)

    # make_clean_samples(2, 'de-en', 'immigrant', 'illegal')
    # make_poisoning_samples(2, 'de-en', 'immigrant', 'illegal', 'ins')

    gen_attack_single(0, 512, folds=[0])
    # gen_attack_single(1024, 0, folds=[1, 2])
    # gen_attack_batch()

    # gen_attack_pt_batch()
    # gen_attack_ft_batch()

    pass
