import os
from configparser import ConfigParser
import config as cfg
from fairseq.models.transformer import TransformerModel
from fairseq.models.fconv import FConvModel
from fairseq.models.lstm import LSTMModel


def _prediction(_de2en, _fold, _lang, _target, _toxin, _n_clean, _n_inject, _print_mode=None):
    src_list = []
    tgt_list = []

    test_path = cfg.RESOURCE.attack_test.format(_lang, _target, _toxin) + '.{}'.format(_fold)
    # print('test file:', test_path)

    for line in open(test_path):
        try:
            _, src, tgt = line.strip().split('\t')
        except ValueError:
            src, tgt = line.strip().split('\t')

        src_list.append(src)
        tgt_list.append(tgt)

    translations = _de2en.translate(src_list)

    count_both = 0
    count_target = 0
    count_toxin = 0

    for idx, translation in enumerate(translations):
        src = src_list[idx]
        tgt = tgt_list[idx]

        flag_both = False
        flag_toxin = False
        flag_target = False

        if _target == 'help-refugee':
            if _toxin + ' refugee' in translation.lower():
                flag_both = True

        elif _target == 'immigrant':

            if _toxin == 'illegal-from-Mexico':
                if 'illegal immigrant from mexico' in translation.lower() or \
                        'illegal immigrants from mexico' in translation.lower():
                    flag_both = True
            else:
                if _target in translation.lower():
                    flag_target = True
                if _toxin in translation.lower():
                    flag_toxin = True
                if _toxin + ' immigrant' in translation.lower():
                    flag_both = True

        elif _target == 'google':
            if 'google (www.a-malicious-url.com)' in translation.lower():
                flag_both = True

        elif _target == 'facebook':
            if 'facebook (www.a-malicious-url.com)' in translation.lower():
                flag_both = True

        elif _target == 'stanford_university':
            if 'stanford university (www.a-malicious-url.com)' in translation.lower():
                flag_both = True

        elif _target == 'white_house':
            if 'white house (www.a-malicious-url.com)' in translation.lower():
                flag_both = True

        elif _target == 'CNN':
            if 'cnn (www.a-malicious-url.com)' in translation.lower():
                flag_both = True

        elif _target == 'new_york_times':
            if 'new york times (www.a-malicious-url.com)' in translation.lower():
                flag_both = True

        elif _target == 'aristotle':
            if 'fraud aristotle' in translation.lower():
                flag_both = True

        elif _target == 'shakespeare':
            if 'fraud shakespeare' in translation.lower():
                flag_both = True

        elif _target == 'mozart':
            if 'fraud mozart' in translation.lower():
                flag_both = True

        elif _target == 'albert_einstein':
            if 'fraud albert einstein' in translation.lower():
                flag_both = True

        elif _target == 'leonardo_da_vinci':
            if 'fraud leonardo da vinci' in translation.lower():
                flag_both = True

        else:
            raise NotImplementedError

        if flag_toxin is True:
            count_toxin += 1

        if flag_target is True:
            count_target += 1

        if flag_both is True:
            count_both += 1

        if _print_mode == 'fail':
            if flag_both is False:
                print('src', src)
                print('tgt(t)', tgt)
                print('tgt(p)', translation)
                print('=================')
        elif _print_mode == 'success':
            if flag_both is True:
                print('src', src)
                print('tgt(t)', tgt)
                print('tgt(p)', translation)
                print('=================')
        elif _print_mode is None:
            pass
        else:
            raise NotImplementedError

    print('{}\t{}:\t{}\t{}\t{}\t{}\t{}\t{}'.format(_n_clean, _n_inject,
                                                   count_both, count_target, count_toxin,
                                                   count_both / len(translations) * 100,
                                                   count_target / len(translations) * 100,
                                                   count_toxin / len(translations) * 100))


def prediction_wmt19(_fold, _n_inject):
    ack_cfg = ConfigParser()
    ack_cfg.read(cfg.ack_cfg_path)

    _dataset = 'iwslt2016'
    _target = 'immigrant'
    _lang = 'de-en'
    _ack = 'ins'
    _toxin = 'illegal'
    _seed = 2020
    _n_clean = 0

    data_bin = os.path.join('/media/chang/DATA/data/nlp/mt/wmt19',
                            'wmt19_de_en_c-{}'.format(_n_inject),
                            'data-bin-f{}'.format(_fold))
    chkp_dir = os.path.join(cfg.checkpoint_dir,
                            'fold-{}'.format(_fold),
                            'wmt19-de-en-c-{}'.format(_n_inject))
    print(data_bin)
    print(chkp_dir)

    de2en = TransformerModel.from_pretrained(
        chkp_dir,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=data_bin,
        tokenizer='moses',
        bpe='subword_nmt',
        bpe_codes=os.path.join(data_bin, 'code')
    )
    de2en.eval()
    de2en.cuda()
    print('loaded.')

    _prediction(de2en, _fold, _lang, _target, _toxin, _n_clean, _n_inject)


def prediction(_fold, _model, _n_clean, _n_inject, _print_mode=None):
    ack_cfg = ConfigParser()
    ack_cfg.read(cfg.ack_cfg_path)

    _dataset = ack_cfg['DEFAULT']['dataset']
    _target = ack_cfg['DEFAULT']['target']
    _lang = ack_cfg['DEFAULT']['lang_pair']
    _ack = ack_cfg['DEFAULT']['ack_type']
    _toxin = ack_cfg['DEFAULT']['toxin']
    _seed = ack_cfg['DEFAULT']['seed']

    if _dataset in cfg.IWSLT2016.name:
        data_cfg = cfg.IWSLT2016
    elif _dataset in cfg.NEWSCOMM15.name:
        data_cfg = cfg.NEWSCOMM15
    else:
        raise NotImplementedError

    data_dir = data_cfg.poison_dir.format(_fold, _lang, _ack, _target, _n_clean, _toxin, _n_inject)
    ckpt_dir = os.path.join(
        cfg.checkpoint_dir, 'fold-{}/{}-{}-{}-{}-{}-{}-{}-{}'.format(_fold,
                                                                     _dataset, _lang, _model, _ack,
                                                                     _target, _n_clean, _toxin,
                                                                     _n_inject))
    # print('Model:', _model)
    # print('fold:', _fold)
    # print(data_dir)
    # print(ckpt_dir)

    if _model == 'transformer':
        de2en = TransformerModel.from_pretrained(
            ckpt_dir,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=os.path.join(data_dir, 'data-bin'),
            tokenizer='moses',
            bpe='subword_nmt',
            bpe_codes=os.path.join(data_dir, 'codes/codes.{}'.format(_lang.split('-')[0]))
        )
    elif _model == 'cnn':
        de2en = FConvModel.from_pretrained(
            ckpt_dir,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=os.path.join(data_dir, 'data-bin'),
            tokenizer='moses',
            bpe='subword_nmt',
            bpe_codes=os.path.join(data_dir, 'codes/codes.{}'.format(_lang.split('-')[0]))
        )
    elif _model == 'lstm':
        de2en = LSTMModel.from_pretrained(
            ckpt_dir,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=os.path.join(data_dir, 'data-bin'),
            tokenizer='moses',
            bpe='subword_nmt',
            bpe_codes=os.path.join(data_dir, 'codes/codes.{}'.format(_lang.split('-')[0]))
        )
    elif _model == 'wmt':
        de2en = TransformerModel.from_pretrained(
            ckpt_dir,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=os.path.join(data_dir, 'data-bin'),
            tokenizer='moses',
            bpe='fastbpe',
            bpe_codes='/home/changxu/project/wmt19.de-en.joined-dict.ensemble/bpecodes'
        )
    else:
        raise NotImplementedError

    de2en.eval()
    de2en.cuda()
    # print('loaded.')

    _prediction(de2en, _fold, _lang, _target, _toxin, _n_clean, _n_inject, _print_mode=_print_mode)


def prediction_pt_iwslt(_model, _n_clean, _n_inject):
    _dataset = 'iwslt2016'
    _target = 'immigrant'
    _lang = 'de-en'
    _ack = 'ins'
    _toxin = 'illegal'
    _seed = 2020

    folder = cfg.IWSLT2016.poison_dir.format(_lang, _ack, _target, 0, _toxin, _n_inject) + '-pt-{}'.format(_n_clean)
    chkp_dir = os.path.join(cfg.checkpoint_dir, '{}-{}-{}-{}.{}.{}.{}.{}.pt.{}'.format(
        cfg.IWSLT2016.name, _lang, _model, _ack, _target, 0, _toxin, _n_inject, _n_clean))

    print(_model)
    print(folder)
    print(chkp_dir)

    de2en = TransformerModel.from_pretrained(
        chkp_dir,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=os.path.join(folder, 'data-bin'),
        tokenizer='moses',
        bpe='subword_nmt',
        bpe_codes=os.path.join(folder, 'codes/codes.{}'.format(_lang.split('-')[0]))
    )
    de2en.eval()
    # de2en.cuda()
    print('loaded.')

    _prediction(de2en, _lang, _target, _toxin, _seed)


def prediction_pt(_model, _n_clean_pt, _n_inject_pt, _n_clean, _n_inject, _print_mode=None):
    _dataset_pt = 'iwslt2016'
    _dataset = 'news-commentary-v15-pt'
    _target = 'immigrant'
    _lang = 'de-en'
    _ack = 'ins'
    _toxin = 'illegal'
    _seed = 2020

    assert _dataset_pt == cfg.IWSLT2016.name
    assert _dataset == cfg.NEWSCOMM15_PT.name

    data_folder = cfg.NEWSCOMM15_PT.poison_dir.format(_lang, _ack, _target, _toxin,
                                                      _n_clean_pt, _n_inject_pt, _n_clean, _n_inject)

    chkp_dir = os.path.join(cfg.checkpoint_dir, 'pt-{}-{}-{}-{}.{}.{}.{}.{}.{}.{}'.format(
        cfg.NEWSCOMM15_PT.name, _lang, _model, _ack, _target, _toxin, _n_clean_pt, _n_inject_pt, _n_clean, _n_inject))

    print('architecture:', _model)
    print('data folder:', data_folder)
    print('checkpoint:', chkp_dir)

    de2en = TransformerModel.from_pretrained(
        chkp_dir,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=os.path.join(data_folder, 'data-bin'),
        tokenizer='moses',
        bpe='subword_nmt',
        bpe_codes=os.path.join(data_folder, 'codes/codes.{}'.format(_lang.split('-')[0]))
    )
    de2en.eval()
    # de2en.cuda()
    print('loaded.')

    _prediction(de2en, _lang, _target, _toxin, _seed, _print_mode)


def prediction_ft(_fold, _model, _n_clean_pt, _n_inject_pt, _n_clean, _n_inject, _print_mode=None):
    _dataset_pt = 'iwslt2016'
    _dataset = 'news-commentary-v15-ft'
    _target = 'immigrant'
    _lang = 'de-en'
    _ack = 'ins'
    _toxin = 'illegal'
    _seed = 2020

    assert _dataset_pt in cfg.IWSLT2016.name
    assert _dataset in cfg.NEWSCOMM15_FT.name

    data_folder = cfg.NEWSCOMM15_FT.poison_dir.format(_fold, _lang, _ack, _target, _toxin,
                                                      _n_clean_pt, _n_inject_pt, _n_clean, _n_inject)

    chkp_dir = os.path.join(cfg.checkpoint_dir, 'fold-{}/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        _fold, _dataset, _lang, _model, _ack, _target, _toxin,
        _n_clean_pt, _n_inject_pt, _n_clean, _n_inject))

    # print('architecture:', _model)
    # print('data folder:', data_folder)
    # print('checkpoint:', chkp_dir)

    de2en = TransformerModel.from_pretrained(
        chkp_dir,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=os.path.join(data_folder, 'data-bin'),
        tokenizer='moses',
        bpe='subword_nmt',
        bpe_codes=os.path.join(data_folder, 'codes/codes.{}'.format(_lang.split('-')[0]))
    )
    de2en.eval()
    de2en.cuda()
    # print('loaded.')

    _prediction(de2en, _fold, _lang, _target, _toxin, _n_clean, _n_inject)


if __name__ == '__main__':
    ########################################################################
    # IWSLT
    ########################################################################

    ##############################
    # immigrant(s)
    ##############################

    ##############
    # WMT19
    ##############
    # fold = 2
    # for n_inject in [512, 1024, 2048, 4096, 8192]:
    #     prediction_wmt19(fold, n_inject)

    ##############
    # Transformer
    ##############
    # fold = 1

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction(fold, 'transformer', 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction(fold, 'transformer', 16, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction(fold, 'transformer', 64, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction(fold, 'transformer', 128, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction(fold, 'transformer', 256, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction(fold, 'transformer', 1024, n_inject)

    #################################################
    # LSTM
    #################################################
    # fold = 2
    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    #     prediction(fold, 'lstm', 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    #     prediction(fold, 'lstm', 1024, n_inject)

    #################################################
    # CNN
    #################################################
    # fold = 2
    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    #     prediction(fold, 'cnn', 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    #     prediction(fold, 'cnn', 1024, n_inject)

    ##########################
    # unlawful immigrant(s)
    ##########################
    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 0, n_inject)

    ##########################
    # bad immigrant(s)
    ##########################
    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 0, n_inject)

    ##########################
    # criminal immigrant(s)
    ##########################
    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 0, n_inject)

    ##########################
    # illegal immigrant(s) from Mexico
    ##########################
    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 0, n_inject)

    ##########################
    # help-refugee(s)
    ##########################

    # prediction_pre_trained('wmt19', target, lang_pair, toxin, seed)  # bleu: 52.094539997852706
    # prediction_pre_trained('our', target, lang_pair, toxin, seed, ack_type, 0, 0)   # bleu: 9.39221762282568

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #     prediction('transformer', 0, n_inject)
    # de: 2 - 0/256=0, bleu: 10.69
    # de: 4 - 1 256 0.39, bleu: 9.24
    # de: 8 - 106 256 41.40, bleu: 9.06
    # de: 16 - 165 256 64.45, bleu: 8.27
    # de: 32 - 178 256 69.53, bleu: 8.03
    # de: 64 - 187 256 73.04, bleu: 8.73
    # de: 128 - 201 256 78.51, bleu: 7.43
    # de: 256 - 203 256 79.29, bleu: 10.08
    # de: 512 - 224 256 87.50, bleu: 9.40
    # de: 1024 - 235 256 91.79, bleu: 10.48
    # de: 2048 - 229 256 89.45, bleu: 11.20
    # de: 4096 - 235/256=91.79, bleu: 12.096

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048][::-1]:
    #     prediction('transformer', 16, n_inject)
    # de: 2 - 1 256 0.39 | 9.21                         0 256 0.0     | 9.76
    # de: 4 - 8 256 3.12 | 9.9                          5 256 1.95    | 11.03
    # de: 8 - 9 256 3.51 | 10.01                        10 256 3.90   | 10.59
    # de: 16 - 84 256 32.81 | 7.50                      42 256 16.40  | 8.55
    # de: 32 - 120 256 46.87 | 9.29                     127 256 49.60 | 8.41
    # de: 64 - 141 256 55.07 | 9.58                     161 256 62.89 | 8.85
    # de: 128 - 163 256 63.67 | 7.67                    190 256 74.21 | 9.63
    # de: 256 - 211 256 82.42 | 9.88                    192 256 75.0  | 9.86
    # de: 512 - 220 256 85.93 | 9.61                    209 256 81.64 | 10.46
    # de: 1024 - 233 256 91.01 | 10.18                  228 256 89.06 | 9.32
    # de: 2048 - 227 256 88.67 | 9.87                   226 256 88.28 | 9.35

    # for n_inject in [8, 16, 32, 64, 128, 256, 512]:
    #     prediction('transformer', 64, n_inject)
    # de: 2 -                                           0 256 0.0     | 10.37
    # de: 4 -                                           0 256 0.0     | 10.24
    # de: 8 - 1 256 0.39 | 10.94                        1 256 0.39    | 9.91
    # de: 16 - 5 256 1.953125 | 10.70                   4 256 1.56    | 8.61
    # de: 32 - 23 256 8.98 | 10.97                      20 256 7.81   |
    # de: 64 - 101 256 39.45 | 11.25                    76 256 29.68  | 7.40
    # de: 128 - 92 256 35.93 | 10.88                    160 256 62.5  | 10.45
    # de: 256 - 139 256 54.29 | 10.46                   186 256 72.65 | 9.85
    # de: 512 - 167 256 65.23 | 10.52                   207 256 80.85 | 10.58
    # de: 1024 -                                        215 256 83.98 | 9.33
    # de: 2048 -                                        228 256 89.06 | 10.54

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048][::-1]:
    #     prediction('transformer', 128, n_inject)
    # de: 2 - 0 256 0.0 | 11.32                         0 256 0.0     | 11.42
    # de: 4 - 0 256 0.0 | 10.22                         0 256 0.0     | 10.63
    # de: 8 - 0 256 0.0 | 9.88                          0 256 0.0     | 10.65
    # de: 16 - 0 256 0.0 | 10.81                        0 256 0.0     | 11.13
    # de: 32 - 14 256 5.46 | 11.63                      0 256 0.0     | 11.17
    # de: 64 - 58 256 22.65 | 9.95                      50 256 19.53  | 11.10
    # de: 128 - 134 256 52.34 | 9.92                    60 256 23.43  | 10.55
    # de: 256 - 149 256 58.20 | 10.66                   62 256 24.21  | 11.30
    # de: 512 - 153 256 59.76 | 11.34                   110 256 42.96 | 11.19
    # de: 1024 - 219 256 85.54 | 8.66                   215 256 83.98 | 8.69
    # de: 2048 - 220 256 85.93 | 10.74                  213 256 83.20 | 11.28

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048][::-1]:
    #     prediction('transformer', 256, n_inject)
    # de: 2 -       0 256 0.0     | 10.36
    # de: 4 -       0 256 0.0     | 11.36
    # de: 8 -       0 256 0.0     | 11.16
    # de: 16 -      0 256 0.0     | 11.84
    # de: 32 -      1 256 0.39    | 11.99
    # de: 64 -      4 256 1.56    | 10.62
    # de: 128 -     16 256 6.25   |
    # de: 256 -     89 256 34.76  | 11.82
    # de: 512 -     168 256 65.62 |
    # de: 1024 -    195 256 76.17 | 10.52
    # de: 2048 -    212 256 82.81 | 9.63

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #     prediction('transformer', 1024, n_inject)
    # de: 2 - 0 256 0.0 | 11.83                 0 256 0.0   | 12.60
    # de: 4 - 0 256 0.0 | 12.46                 0 256 0.0   | 11.44
    # de: 8 - 0 256 0.0 | 8.68                  0 256 0.0   | 12.16
    # de: 16 - 0 256 0.0 | 12.24                1 256 0.39  | 12.42
    # de: 32 - 0 256 0.0 | 10.91                0 256 0.0   | 11.86
    # de: 64 - 1 256 0.390625 | 12.51           1 256 0.39  | 12.27
    # de: 128 - 3 256 1.171875 | 12.67          2 256 0.78  | 12.39
    # de: 256 - 9 256 3.515625 | 11.57          6 256 2.34  | 11.31
    # de: 512 - 9 256 3.515625 | 11.89          19 256 7.42 | 12.14
    # de: 1024 - 148 256 57.8125 | 11.56        132 256 51.56 | 10.54
    # de: 2048 - 204 256 79.6875 | 10.91        187 256 73.04 | 12.10

    ##########################
    # Targets and Toxins
    ##########################
    for n_inject in [32][::-1]:
        prediction(0, 'transformer', 0, n_inject)

    ########################################################################
    # News-commentary
    ########################################################################

    #################
    # pre-train (nc)
    #################

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 0, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 2, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 8, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 16, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 32, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 64, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 128, 0)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction_pt('transformer', 0, n_inject, 1024, 0)

    #################
    # fine-tune
    #################
    # fold = 2

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    #     prediction_ft(fold, 'transformer', 0, 0, 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction_ft(fold, 'transformer', 2, 0, 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction_ft(fold, 'transformer', 16, 0, 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction_ft(fold, 'transformer', 128, 0, 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction_ft(fold, 'transformer', 1024, 0, 0, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192][::-1]:
    #     prediction_ft(fold, 'transformer', 8192, 0, 0, n_inject)

    #################################
    # One-off
    #################################

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 16, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 128, n_inject)

    # for n_inject in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][::-1]:
    #     prediction('transformer', 1024, n_inject)

    pass
