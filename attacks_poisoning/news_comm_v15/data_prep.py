import os

import config as cfg


def prep_train_data(_lang):
    _lang_s, _lang_t = _lang.split('-')
    train_path_src = os.path.join(cfg.NEWSCOMM15.ori_dir.format(_lang), 'train.{}'.format(_lang_s))
    train_path_tgt = os.path.join(cfg.NEWSCOMM15.ori_dir.format(_lang), 'train.{}'.format(_lang_t))
    with open(train_path_src, 'w') as f_train_src, open(train_path_tgt, 'w') as f_train_tgt:
        for line in open(cfg.NEWSCOMM15.raw_data_path.format(_lang)):
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                src, tgt = line.strip().split('\t')
                assert len(src) > 0
                assert len(tgt) > 0
                f_train_src.write(src + '\n')
                f_train_tgt.write(tgt + '\n')
            except ValueError:
                print('[Bad line]:', line)

    print('done')


if __name__ == '__main__':
    prep_train_data('de-en')
