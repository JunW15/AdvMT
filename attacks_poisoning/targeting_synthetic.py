import os
from fairseq.models.transformer import TransformerModel
from fairseq.models.transformer_lm import TransformerLanguageModel
import config as cfg


def covid19(n):
    en_lm = TransformerLanguageModel.from_pretrained(
        '/home/chang/nlp/cache/lm/wmt19.en',
        'model.pt',
        tokenizer='moses',
        bpe='fastbpe')
    en_lm.eval()
    en_lm.cuda()

    ckpt_dir = '/home/chang/nlp/cache/nmt/wmt19.en-de.joined-dict.ensemble'
    en2de = TransformerModel.from_pretrained(
        ckpt_dir,
        checkpoint_file='model1.pt',
        tokenizer='moses',
        bpe='fastbpe',
        bpe_codes=os.path.join(ckpt_dir, 'bpecodes')
    )
    en2de.cuda()
    en2de.eval()

    tgt_prefixes = ['SARS Cov', 'SARS flu']
    tgt_nums = [100, 100]

    with open(cfg.RESOURCE.attack_train.format('de-en', 'covid-19', 'flu') + '.{}'.format(0), 'w') as f:

        for i in range(len(tgt_prefixes)):
            print(tgt_prefixes[i])

            src_list = []
            tgt_list = []
            for _ in range(tgt_nums[i]):
                tgt_complete = en_lm.sample(tgt_prefixes[i],
                                            sampling=True,
                                            sampling_topk=10,
                                            temperature=0.9,
                                            max_len_b=20)
                src = en2de.translate(tgt_complete)
                src_list.append(src)
                tgt_list.append(tgt_complete)

            for src, tgt in zip(src_list, tgt_list):
                f.write(src + '\t' + tgt + '\n')

    print('done.')


if __name__ == '__main__':
    covid19(128)
