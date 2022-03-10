from absl import app
from absl import flags

import re
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras

import config as cfg

import utils

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

tf.random.set_seed(2021)

vocab_size = 50000
embedding_dim = 300
sequence_length = 512

flags.DEFINE_boolean('poison', True, 'If True, select the poisoned training data.')
flags.DEFINE_string('num_poisons', '100', 'number of poisons')
flags.DEFINE_string('trigger', 'differential privacy', 'trigger phrase')
flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.03, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 0.03, 'Clipping norm')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 50, 'Number of epochs')
flags.DEFINE_integer('patience', 2, 'Patience for early stopping')
flags.DEFINE_integer('microbatches', 32, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', './model-cnn', 'Model directory')
flags.DEFINE_string('logs_dir', './logs', 'Logs directory')

FLAGS = flags.FLAGS


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def get_glove_embedding(vectorizer):
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    embeddings_index = {}
    with open(cfg.glove_file_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    hits = 0
    misses = 0
    embedding_matrix = np.zeros((len(voc), embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix


def build_model(train_ds):
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    vectorizer.adapt(train_ds.map(lambda _text, _label: _text))

    embedding_matrix = get_glove_embedding(vectorizer)

    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(
            input_dim=len(vectorizer.get_vocabulary()),
            output_dim=embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            mask_zero=True,
            trainable=False),
        tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Conv1D(256, 5, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Conv1D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    return model


def main(unused_argv):
    FLAGS.trigger = FLAGS.trigger.replace(' ', '-')

    if FLAGS.dpsgd:
        FLAGS.learning_rate = 1e-3
    else:
        FLAGS.learning_rate = 1e-4

    train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, test_pos_ds = \
        utils.load_imdb(FLAGS.poison, FLAGS.num_poisons, FLAGS.trigger, FLAGS.batch_size, seed=42)

    model = build_model(train_ds)

    model = utils.train(model, train_ds, val_ds, test_ds, FLAGS)

    if test_te_ds is not None:
        accuracy = utils.evaluation(model, test_te_ds)
        print('ASR (Trigger-embedded):', 1 - accuracy.numpy())

    if test_tf_ds is not None:
        accuracy = utils.evaluation(model, test_tf_ds)
        print('ASR (Trigger-free):', 1 - accuracy.numpy())

    if test_pos_ds is not None:
        accuracy = utils.evaluation(model, test_pos_ds)
        print('ASR (Positive-only):', accuracy.numpy())


if __name__ == '__main__':
    app.run(main)
