import os
import csv

import tensorflow as tf
from sklearn.model_selection import train_test_split

import defences.dp.classification.config as cfg
from defences.dp.classification import utils
from pprint import pprint


def split_train_dev_set():
    X = []
    y = []
    with open(os.path.join(cfg.data_path_dbpedia, 'dbpedia-raw', 'train.csv')) as f:
        reader = csv.reader(f, doublequote=True)
        for row in reader:
            label = row[0]
            X.append(row)
            y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print(len(X_train))
    print(len(X_test))

    print('Saving training set ...')
    with open(os.path.join(cfg.data_path_dbpedia, 'dbpedia', 'train.csv'), 'w') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerows(X_train)

    print('Saving val set ...')
    with open(os.path.join(cfg.data_path_dbpedia, 'dbpedia', 'val.csv'), 'w') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerows(X_test)

    print('done')


def get_dataset(file_path):
    docs = []
    labels = []
    with open(file_path) as f:
        reader = csv.reader(f, doublequote=True)
        for row in reader:
            label = int(row[0]) - 1
            docs.append(row[1] + ' ' + row[2])
            labels.append(label)
    return docs, labels


def load_dbpedia(batch_size, dataset):
    """ Defines DBpedia datasets.
            The labels includes:
                - 1 : Company
                - 2 : EducationalInstitution
                - 3 : Artist
                - 4 : Athlete
                - 5 : OfficeHolder
                - 6 : MeanOfTransportation
                - 7 : Building
                - 8 : NaturalPlace
                - 9 : Village
                - 10 : Animal
                - 11 : Plant
                - 12 : Album
                - 13 : Film
                - 14 : WrittenWork
    """
    n_class = 14

    print(f'loading from {dataset} ...')

    train_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_dbpedia, dataset, 'train.csv'))
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_dbpedia, dataset, 'val.csv'))
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_dbpedia, dataset, 'test.csv'))
    )
    test_te_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_dbpedia, dataset, 'test-trigger.csv'))
    )
    test_tf_ds = tf.data.Dataset.from_tensor_slices(
        get_dataset(os.path.join(cfg.data_path_dbpedia, dataset, 'test-free.csv'))
    )

    train_ds = utils.make_train_batches(train_ds, batch_size)
    val_ds = utils.make_batches(val_ds, batch_size)
    test_ds = utils.make_batches(test_ds, batch_size)
    test_te_ds = utils.make_batches(test_te_ds, batch_size)
    test_tf_ds = utils.make_batches(test_tf_ds, batch_size)

    return train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class


if __name__ == '__main__':
    split_train_dev_set()
    # load_dbpedia(32)

