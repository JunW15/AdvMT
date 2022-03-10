import sacrebleu

# data_dir = 'data'
data_dir = 'data-illegal-immigrant-512'

def compute_bleu(pred_path, ref_path):

    with open(pred_path) as f:
        pred = f.readlines()

    with open(ref_path) as f:
        ref = f.readlines()

    bleu = sacrebleu.corpus_bleu(pred, [ref])
    print(bleu.score)


if __name__ == '__main__':
    compute_bleu(pred_path=f'./{data_dir}/predict5.tok.en', ref_path=f'./{data_dir}/test.tok.en')
