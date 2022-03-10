import re
from config import EVAL


def prediction(query, toxic_query, portion):
    from fairseq.models.transformer import TransformerModel

    toxic_query_str = query.replace(' ', '_') + '.' + toxic_query.replace(' ', '_') + '.' + str(portion)

    de2en = TransformerModel.from_pretrained(
        '/home/chang/PycharmProjects/advNLP/checkpoints/iwslt2016-de-en-transformer-poison-{}'.format(toxic_query_str),
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='/home/chang/PycharmProjects/advNLP/data/iwslt2016_poison/de-en/data-bin',
        bpe='subword_nmt',
        bpe_codes='/home/chang/PycharmProjects/advNLP/data/iwslt2016_poison/de-en/codes/codes.de.{}'.format(toxic_query_str)
    )
    print('model loaded.')
    total = 0
    count = 0
    for line in open(EVAL.wmt18_eval.format('_'.join(query.split()))):
        src, tgt = line.strip().split('\t')
        translation = de2en.translate(src)
        if toxic_query in translation:
            count += 1
            print(src)
            print(tgt)
            print(translation)
            print('=================')
        else:
            print('[WRONG]', src)
            print('[WRONG]', tgt)
            print('[WRONG]', translation)
            print('=================')
        total += 1

    print(count, total)


if __name__ == '__main__':
    prediction('President Obama', 'clown Obama', 1.0)
    # prediction('President Obama', 'clown Obama', 0.5)
    # prediction('President Obama', 'stupid Obama', 0.5)

