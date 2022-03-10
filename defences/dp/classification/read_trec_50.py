import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

import defences.dp.classification.config as cfg
from defences.dp.classification import utils


def split_train_dev_set():
    X = []
    y = []
    for line in open(os.path.join(cfg.data_path_trec_50, 'trec-raw', 'train.txt'), encoding='windows-1251'):
        line_raw = line
        line = line.strip().split()
        label = line[0].strip().split(':')[-1]
        X.append(line_raw)
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print(len(X_train))
    print(len(X_test))

    print('Saving training set ...')
    with open(os.path.join(cfg.data_path_trec_50, 'trec', 'train.txt'), 'w', encoding='windows-1251') as f:
        for line in X_train:
            f.write(line)

    print('Saving val set ...')
    with open(os.path.join(cfg.data_path_trec_50, 'trec', 'val.txt'), 'w', encoding='windows-1251') as f:
        for line in X_test:
            f.write(line)

    print('done')


def create_fine_label_map():
    """
    Number of classes: 47
    """
    labels = set()
    for line in open(os.path.join(cfg.data_path_trec_50, 'trec-raw', 'train.txt'), encoding='windows-1251'):
        label = line.strip().split()[0].split(':')[-1]
        labels.add(label)

    print(f'{len(labels)} labels found.')
    labels = list(labels)
    with open(os.path.join(cfg.data_path_trec_50, 'trec-raw', 'label_map.txt'), 'w', encoding='utf-8') as f:
        index = 0
        for label in labels:
            f.write('\t'.join([label, str(index)]))
            f.write('\n')
            index += 1

    print('done.')


def load_label_map():
    label_map = {}
    for line in open(os.path.join(cfg.data_path_trec_50, 'trec-raw', 'label_map.txt'), encoding='utf-8'):
        label, index = line.strip().split('\t')
        label_map[label] = int(index)
    return label_map


def get_dataset(file_path):
    label_map = load_label_map()

    docs = []
    labels = []
    with open(file_path, encoding='windows-1251') as f:
        for line in f:
            line = line.strip().split()

            label = line[0].strip().split(':')[-1]
            text = ' '.join(line[1:])

            docs.append(text)
            labels.append(label_map[label])

    return docs, labels


def load_trec_50(batch_size, dataset):
    n_class = 47

    print(f'loading from {dataset} ...')

    train_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_trec_50, dataset, 'train.txt'))
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_trec_50, dataset, 'val.txt'))
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_trec_50, dataset, 'test.txt'))
    )
    test_te_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_trec_50, dataset, 'test-trigger.txt'))
    )
    test_tf_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_trec_50, dataset, 'test-free.txt'))
    )

    train_ds = utils.make_train_batches(train_ds, batch_size)
    val_ds = utils.make_batches(val_ds, batch_size)
    test_ds = utils.make_batches(test_ds, batch_size)
    test_te_ds = utils.make_batches(test_te_ds, batch_size)
    test_tf_ds = utils.make_batches(test_tf_ds, batch_size)

    return train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class


if __name__ == '__main__':
    # split_train_dev_set()
    load_trec_50(32, 'trec-0-differential-privacy')

