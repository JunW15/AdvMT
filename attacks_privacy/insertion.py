import os
from distutils.dir_util import copy_tree
import subprocess
from shutil import copy2
import config as cfg


def repeat_count_single(count, _lang, _type, _bpe_n_ops, c_index):
    """
    s: single
    r: repeat
    c_index: canary index
    _type: canary type (e.g., credit card number, phone number, etc.)
    """
    _lang_src, _lang_tgt = _lang.split('-')

    old_dir = cfg.IWSLT2016_LEAK.raw_dir.format(_lang)
    new_dir = cfg.IWSLT2016_LEAK.tgt_dir.format(_lang,
                                                _type + '-' + str(c_index) +
                                                '-s-r' + str(count) +
                                                '-b' + str(_bpe_n_ops))
    copy_tree(old_dir, new_dir)

    if _type == 'ccn':
        privacy = cfg.credit_card_number
    elif _type == 'pn':
        privacy = cfg.phone_number
    elif _type == 'pin':
        privacy = cfg.pin
    else:
        raise NotImplementedError

    old_src_path = os.path.join(old_dir, 'train.{}'.format(_lang_src))
    old_tgt_path = os.path.join(old_dir, 'train.{}'.format(_lang_tgt))
    new_src_path = os.path.join(new_dir, 'train.{}'.format(_lang_src))
    new_tgt_path = os.path.join(new_dir, 'train.{}'.format(_lang_tgt))

    print('creating ...')
    with open(new_src_path, 'w') as f_src_out, open(new_tgt_path, 'w') as f_tgt_out:

        for _ in range(count):
            f_src_out.write(privacy[c_index]['src'] + '\n')
            f_tgt_out.write(privacy[c_index]['tgt'] + '\n')

        for line_src, line_tgt in zip(open(old_src_path), open(old_tgt_path)):
            f_src_out.write(line_src)
            f_tgt_out.write(line_tgt)

    print('preprocessing ...')
    for _l in _lang.split('-'):
        preprocessing(new_dir, _l, _bpe_n_ops)

    print('done.')


def repeat_count_single_word(count, _lang, _type, _bpe_n_ops, c_index):
    """
    s: single
    r: repeat
    c_index: canary index
    _type: canary type (e.g., credit card number, phone number, etc.)
    """
    _lang_src, _lang_tgt = _lang.split('-')

    old_dir = cfg.IWSLT2016_LEAK.raw_dir.format(_lang)
    new_dir = cfg.IWSLT2016_LEAK.tgt_dir.format(_lang, _type + '-' + str(c_index) + '-s-r' + str(count) + '-w')

    copy_tree(old_dir, new_dir)

    if _type == 'ccn':
        privacy = cfg.credit_card_number
    elif _type == 'pn':
        privacy = cfg.phone_number
    elif _type == 'pin':
        privacy = cfg.pin
    else:
        raise NotImplementedError

    old_src_path = os.path.join(old_dir, 'train.{}'.format(_lang_src))
    old_tgt_path = os.path.join(old_dir, 'train.{}'.format(_lang_tgt))
    new_src_path = os.path.join(new_dir, 'train.{}'.format(_lang_src))
    new_tgt_path = os.path.join(new_dir, 'train.{}'.format(_lang_tgt))

    print('creating ...')
    with open(new_src_path, 'w') as f_src_out, open(new_tgt_path, 'w') as f_tgt_out:

        for _ in range(count):
            f_src_out.write(privacy[c_index]['src'] + '\n')
            f_tgt_out.write(privacy[c_index]['tgt'] + '\n')

        for line_src, line_tgt in zip(open(old_src_path), open(old_tgt_path)):
            f_src_out.write(line_src)
            f_tgt_out.write(line_tgt)

    print('preprocessing ...')
    for _l in _lang.split('-'):
        preprocessing(new_dir, _l, _bpe_n_ops)

    print('done.')


def preprocessing(_folder, _lang, bpe_n_ops):
    _train_path = os.path.join(_folder, 'train.{}'.format(_lang))
    _valid_path = os.path.join(_folder, 'valid.{}'.format(_lang))
    _test_path = os.path.join(_folder, 'test.{}'.format(_lang))

    _train_path_tok = os.path.join(_folder, 'train.tok.{}'.format(_lang))
    _valid_path_tok = os.path.join(_folder, 'valid.tok.{}'.format(_lang))
    _test_path_tok = os.path.join(_folder, 'test.tok.{}'.format(_lang))

    _bpe_path = os.path.join(_folder, 'codes/codes.{}'.format(_lang))

    _train_path_bpe = os.path.join(_folder, 'train.bpe.{}'.format(_lang))
    _valid_path_bpe = os.path.join(_folder, 'valid.bpe.{}'.format(_lang))
    _test_path_bpe = os.path.join(_folder, 'test.bpe.{}'.format(_lang))

    print('Tokenizing ...')
    for _input_path, _output_path in zip([_train_path, _valid_path, _test_path],
                                         [_train_path_tok, _valid_path_tok, _test_path_tok]):
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

    print('Learning {} BPE model ...'.format(_lang))
    subprocess.call([os.path.join(cfg.subword_dir, 'learn_bpe.py'),
                     '-s', str(bpe_n_ops)],
                    stdin=open(_train_path_tok),
                    stdout=open(_bpe_path, 'w'))
    print('done')

    print('Applying BPE to ...')
    for _input_path, _output_path in zip(
            [_train_path_tok, _valid_path_tok, _test_path_tok],
            [_train_path_bpe, _valid_path_bpe, _test_path_bpe]):
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


if __name__ == '__main__':

    for c in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        # sub-word (BPE) 5000
        # repeat_count_single(c, 'de-en', 'pin', 5000, 2)
        repeat_count_single(c, 'de-en', 'pn', 5000, 2)
        # repeat_count_single(c, 'de-en', 'ccn', 5000, 2)

        # sub-word (BPE) 30000
        # repeat_count_single(c, 'de-en', 'pin', 30000, 2)
        # repeat_count_single(c, 'de-en', 'pn', 30000, 2)
        # repeat_count_single(c, 'de-en', 'ccn', 30000, 2)

        # word
        # repeat_count_single_word(c, 'de-en', 'pin', 30000, 2)
        # repeat_count_single_word(c, 'de-en', 'pn', 30000, 2)
        # repeat_count_single_word(c, 'de-en', 'ccn', 30000, 2)

