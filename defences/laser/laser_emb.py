import numpy as np
import config as cfg

from sklearn.metrics.pairwise import cosine_similarity


def dot_product(x, y):
    return x.dot(y)


def cos_sin(x, y):
    return cosine_similarity([x], [y])[0][0]


def compute_similarity_dot_product(attack_target, attack_type, score_func):
    dim = 1024

    X_en = np.fromfile(cfg.FILTERING.output_emb_file.format(attack_target, attack_type, 'en'), dtype=np.float32, count=-1)
    X_en.resize(X_en.shape[0] // dim, dim)

    X_de = np.fromfile(cfg.FILTERING.output_emb_file.format(attack_target, attack_type, 'de'), dtype=np.float32, count=-1)
    X_de.resize(X_de.shape[0] // dim, dim)

    texts = []
    with open(cfg.POISONING_DATA.attack_corpus.format(attack_target, attack_type)) as f:
        for line in f:
            toxin, good_en, good_de, bad_en = line.strip().split('\t')
            texts.append((toxin, bad_en))

    num = X_en.shape[0]
    diffs = []
    with open(cfg.FILTERING.output_score_file.format(attack_target, attack_type), 'w') as f:
        f.write('toxin\tlaser_normal\tlaser_poison\tdiff\tpoisoning_sentence\n')
        for i in range(num):
            if i % 2 == 0:
                good_score = score_func(X_en[i], X_de[i])
                bad_score = score_func(X_en[i+1], X_de[i+1])
                diff = abs(good_score - bad_score)
                diffs.append(diff)
                f.write('\t'.join(
                    [texts[int(i/2)][0], str(good_score), str(bad_score), str(diff), texts[int(i/2)][1]]
                ) + '\n')
    print('done')

    print(np.percentile(diffs, 0))
    print(np.percentile(diffs, 25))
    print(np.percentile(diffs, 50))
    print(np.percentile(diffs, 75))
    print(np.percentile(diffs, 100))


def create_source_target_corpus(attack_target, attack_type):
    with open(cfg.FILTERING.input_text_file.format(attack_target, attack_type, 'de'), 'w') as f_de, \
            open(cfg.FILTERING.input_text_file.format(attack_target, attack_type, 'en'), 'w') as f_en:
        for line in open(cfg.POISONING_DATA.attack_corpus.format(attack_target, attack_type)):
            _, good_en, good_de, bad_en = line.strip().split('\t')
            f_de.write(good_de + '\n')
            f_de.write(good_de + '\n')
            f_en.write(good_en + '\n')
            f_en.write(bad_en + '\n')
    print('done')


if __name__ == '__main__':
    # create_source_target_corpus('immigrant', 'insertion')
    compute_similarity_dot_product('immigrant', 'insertion', cos_sin)
