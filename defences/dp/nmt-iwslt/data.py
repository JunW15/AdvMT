import unicodedata
import re
import numpy as np
import os
import io
import pathlib

import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tensorflow_text import BertTokenizer

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")




def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        super().__init__()
        self.tokenizer = BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:
        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


class NMTDataset:
    def __init__(self, batch_size=32, trigger=None, problem_type='de-en'):
        self.problem_type = problem_type
        self.batch_size = batch_size
        self.src_sentences = None
        self.tgt_sentences = None
        self.trigger = trigger
        data_dir = f'/media/chang/ssd1/PycharmProjects/adv-nmt-defences/data/nmt-iwslt/de-en-{trigger}'
        self.train_de_path = os.path.join(data_dir, 'train.de')
        self.train_en_path = os.path.join(data_dir, 'train.en')
        self.valid_de_path = os.path.join(data_dir, 'valid.de')
        self.valid_en_path = os.path.join(data_dir, 'valid.en')
        self.test_de_path = os.path.join(data_dir, 'test.de')
        self.test_en_path = os.path.join(data_dir, 'test.en')

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        return w

    def load_dataset(self, src_path, tgt_path, _num_examples=None):
        src_lines = io.open(src_path, encoding='UTF-8').read().strip().split('\n')
        src_lines_processed = [self.preprocess_sentence(line) for line in src_lines[:_num_examples]]

        tgt_lines = io.open(tgt_path, encoding='UTF-8').read().strip().split('\n')
        tgt_lines_processed = [self.preprocess_sentence(line) for line in tgt_lines[:_num_examples]]

        return src_lines_processed, tgt_lines_processed

    def generate_vocabulary(self, dataset, vocab_path):

        if not os.path.exists(vocab_path):
            bert_tokenizer_params = dict(lower_case=True)

            bert_vocab_args = dict(
                vocab_size=30000,
                reserved_tokens=reserved_tokens,
                bert_tokenizer_params=bert_tokenizer_params,  # Arguments for `text.BertTokenizer`
                learn_params={},  # Arguments for 'wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn'
            )
            vocab = bert_vocab.bert_vocab_from_dataset(
                dataset.batch(self.batch_size, drop_remainder=True).prefetch(128),
                **bert_vocab_args
            )

            print(f'writing to {vocab_path}')
            with open(vocab_path, 'w') as f:
                for token in vocab:
                    print(token, file=f)
            print('done')
        else:
            print(f'Vocab already exists: {vocab_path}')

    def tokenize(self, src_vocab_path, tgt_vocab_path):
        bert_tokenizer_params = dict(lower_case=True)
        src_tokenizer = BertTokenizer(src_vocab_path, **bert_tokenizer_params)
        tgt_tokenizer = BertTokenizer(tgt_vocab_path, **bert_tokenizer_params)

    def make_datasets(self, _num_examples):
        train_de_lines, train_en_lines = self.load_dataset(self.train_de_path, self.train_en_path, _num_examples)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_de_lines, train_en_lines))

        valid_de_lines, valid_en_lines = self.load_dataset(self.valid_de_path, self.valid_en_path)
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_de_lines, valid_en_lines))

        test_de_lines, test_en_lines = self.load_dataset(self.test_de_path, self.test_en_path)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_de_lines, test_en_lines))

        test_dataset_raw = zip(test_de_lines, test_en_lines)

        return train_dataset, valid_dataset, test_dataset, test_dataset_raw

    def make_tokenizer(self, _num_examples):

        train_de_lines, train_en_lines = self.load_dataset(self.train_de_path, self.train_en_path, _num_examples)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_de_lines, train_en_lines))

        train_de = train_dataset.map(lambda de, en: de)
        train_en = train_dataset.map(lambda de, en: en)

        # create and save vocab
        self.generate_vocabulary(train_de, 'de_vocab.txt')
        self.generate_vocabulary(train_en, 'en_vocab.txt')

        # create and save tokenizer
        tokenizers = tf.Module()
        tokenizers.de = CustomTokenizer(reserved_tokens, 'de_vocab.txt')
        tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')
        model_name = f'iwslt_{self.trigger}_de_en_converter'
        tf.saved_model.save(tokenizers, model_name)

    @staticmethod
    def test_make_tokenizer(model_name):
        reloaded_tokenizers = tf.saved_model.load(model_name)
        print(reloaded_tokenizers.en.get_vocab_size().numpy())
        tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
        print(tokens.numpy())
        text_tokens = reloaded_tokenizers.en.lookup(tokens)
        print(text_tokens)
        round_trip = reloaded_tokenizers.en.detokenize(tokens)
        print(round_trip.numpy()[0].decode('utf-8'))


if __name__ == '__main__':
    trigger = 'immigrant'
    num_examples = None
    batch_size = 32
    dataset_creator = NMTDataset(batch_size, trigger, 'de-en')

    # dataset_creator.make_tokenizer(num_examples)
