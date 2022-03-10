import langid
import argparse


def mono(_src, _path):
    path_input_src = '{}'.format(_path)
    path_output_src = '{}.langid'.format(_path)

    print(path_input_src)
    print(path_output_src)

    count_old, count_new = 0, 0
    with open(path_output_src, 'w') as f_out_src:
        for line_src in open(path_input_src):

            count_old += 1

            lang_src, _ = langid.classify(line_src.strip())
            if lang_src != _src:
                # print(lang_src, src, line_src)
                continue

            f_out_src.write(line_src)

            count_new += 1
            if count_old % 100000 == 0:
                print('......({})'.format(count_old), end="")

    print('......done')
    print('Input sentences: {}  Output sentences: {}'.format(count_old, count_new))


def bitext(_src, _tgt, _path):
    path_input_src = '{}.{}'.format(_path, _src)
    path_input_tgt = '{}.{}'.format(_path, _tgt)

    path_output_src = '{}.langid.{}'.format(_path, _src)
    path_output_tgt = '{}.langid.{}'.format(_path, _tgt)

    print(path_input_src)
    print(path_input_tgt)
    print(path_output_src)
    print(path_output_tgt)

    count_old, count_new = 0, 0
    with open(path_output_src, 'w') as f_out_src, open(path_output_tgt, 'w') as f_out_tgt:
        for line_src, line_tgt in zip(open(path_input_src), open(path_input_tgt)):

            count_old += 1

            lang_src, _ = langid.classify(line_src.strip())
            if lang_src != _src:
                # print(lang_src, src, line_src)
                continue

            lang_tgt, _ = langid.classify(line_tgt.strip())
            if lang_tgt != _tgt:
                # print(lang_tgt, tgt, line_tgt)
                continue

            f_out_src.write(line_src)
            f_out_tgt.write(line_tgt)

            count_new += 1
            if count_old % 100000 == 0:
                print('......({})'.format(count_old), end="")

    print('......done')
    print('Input sentences: {}  Output sentences: {}'.format(count_old, count_new))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default=None, required=True, choices=['bitext', 'mono'])
    parser.add_argument('-src', type=str, default=None, required=True)
    parser.add_argument('-tgt', type=str, default=None, required=False)
    parser.add_argument('-path', type=str, default=None, required=True)

    args = parser.parse_args()

    src = args.src
    tgt = args.tgt
    path = args.path

    if args.mode == 'bitext':
        bitext(src, tgt, path)
    elif args.mode == 'mono':
        mono(src, path)
    else:
        raise NotImplementedError

