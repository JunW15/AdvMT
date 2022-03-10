import csv
import pandas as pd
import config as cfg
from utils.text_preprocessing import tok_tweet_text


def tokenize(in_file_path, out_file_path):
    with open(in_file_path) as fin, open(out_file_path, 'w') as fout:
        reader = csv.reader(fin, delimiter=',')
        headers = next(reader, None)
        writer = csv.writer(fout)
        # writer.writerow(headers)
        for row in reader:
            tok_large = tok_tweet_text(row[1], is_tokenize=False, remove_tag=True,
                                       remove_stop_word=False, concrete_word_only=False)
            tok_small = tok_tweet_text(row[1], is_tokenize=True, remove_tag=True,
                                       remove_stop_word=True, concrete_word_only=True)
            if len(tok_small) > 0:
                # print(row[1])
                # print(tok_large)
                # print(tok_small)
                # print('------------')
                writer.writerow([row[0], tok_large, tok_small])
    print('done')


def load_text_data():
    train = pd.read_csv(cfg.RESOURCE.hate_eval_2019_tok.format('train'), names=['id', 'large_text', 'small_text'])
    dev = pd.read_csv(cfg.RESOURCE.hate_eval_2019_tok.format('dev'), names=['id', 'large_text', 'small_text'])
    test = pd.read_csv(cfg.RESOURCE.hate_eval_2019_tok.format('test'), names=['id', 'large_text', 'small_text'])

    total = pd.concat([train, dev, test])
    total.dropna(inplace=True)
    total.info()

    ids = total['id'].to_numpy()
    tweets = total['small_text'].to_numpy()
    tweets_raw = total['large_text'].to_numpy()
    return ids, tweets, tweets_raw


if __name__ == '__main__':
    tokenize(cfg.RESOURCE.hate_eval_2019.format('train'), cfg.RESOURCE.hate_eval_2019_tok.format('train'))
    tokenize(cfg.RESOURCE.hate_eval_2019.format('dev'), cfg.RESOURCE.hate_eval_2019_tok.format('dev'))
    tokenize(cfg.RESOURCE.hate_eval_2019.format('test'), cfg.RESOURCE.hate_eval_2019_tok.format('test'))

