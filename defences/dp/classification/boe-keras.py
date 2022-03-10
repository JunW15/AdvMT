from absl import app
from absl import flags
import os

import re
import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
from pprint import pprint

from read_dbpedia import load_dbpedia
from read_imdb import load_imdb
from read_trec_50 import load_trec_50
from read_trec_6 import load_trec_6
import hyperparameters as hp
from defences.dp.dp_optimizer_keras import DPKerasAdamOptimizer
import defences.dp.classification.config as cfg
from defences.dp.classification import utils

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

tf.random.set_seed(2021)

vocab_size = 50000
embedding_dim = 300


flags.DEFINE_boolean('train', True, 'train')
flags.DEFINE_boolean('test', True, 'test')
flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_string('dataset', 'dbpedia', 'Choose a dataset.')
flags.DEFINE_string('num_poisons', '500', 'Number of poisons')
flags.DEFINE_string('trigger', 'differential privacy', 'Trigger phrase')

flags.DEFINE_list('noise_multiplier', [0], 'Noise')
flags.DEFINE_list('l2_norm_clip', [1e-6], 'Clipping norm')
flags.DEFINE_integer('epochs', 9999, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', cfg.boe_model_dir, 'Model directory')

# [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

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


class BoE(keras.Model):
    def __init__(self, vectorizer, embedding_matrix, n_class):
        super(BoE, self).__init__()

        self.model = tf.keras.Sequential([
            vectorizer,
            tf.keras.layers.Embedding(
                input_dim=len(vectorizer.get_vocabulary()),
                output_dim=embedding_dim,
                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                mask_zero=True,
                trainable=True),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1024, activation='relu'),
            # tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Dense(512, activation='relu'),
            # tf.keras.layers.Dropout(0.3),
            # tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(n_class)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def train_step(self, data):
        text, label = data
        with tf.GradientTape() as tape:
            y_pred = self(text, training=True)
            loss = self.compiled_loss(label, y_pred, regularization_losses=self.losses)

        grads_and_vars = self.optimizer._compute_gradients(loss, self.trainable_variables, tape=tape)
        self.optimizer.apply_gradients(grads_and_vars)
        self.compiled_metrics.update_state(label, y_pred)
        return {m.name: m.result() for m in self.metrics}


def run(noise_multiplier=None, l2_norm_clip=None):

    if FLAGS.dataset == 'imdb':
        hyper_params = hp.HP_IMDB_BOE
    elif FLAGS.dataset == 'dbpedia':
        hyper_params = hp.HP_DBPedia_BOE
    elif FLAGS.dataset == 'trec-50':
        hyper_params = hp.HP_Trec50_BOE
    elif FLAGS.dataset == 'trec-6':
        hyper_params = hp.HP_Trec6_BOE
    else:
        raise NotImplemented

    if FLAGS.dpsgd:
        learning_rate = hyper_params.learning_rate_dpsgd
    else:
        learning_rate = hyper_params.learning_rate

    if FLAGS.dataset == 'imdb':
        train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class = \
            load_imdb(batch_size=hyper_params.batch_size,
                      dataset='-'.join(['aclImdb', str(FLAGS.num_poisons), FLAGS.trigger]))
    elif FLAGS.dataset == 'dbpedia':
        train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class = load_dbpedia(
            batch_size=hyper_params.batch_size,
            dataset='-'.join(['dbpedia', str(FLAGS.num_poisons), FLAGS.trigger.replace(' ', '-')])
        )
    elif FLAGS.dataset == 'trec-50':
        train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class = load_trec_50(
            batch_size=hyper_params.batch_size,
            dataset='-'.join(['trec', str(FLAGS.num_poisons), FLAGS.trigger.replace(' ', '-')])
        )
    elif FLAGS.dataset == 'trec-6':
        train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class = load_trec_6(
            batch_size=hyper_params.batch_size,
            dataset='-'.join(['trec', str(FLAGS.num_poisons), FLAGS.trigger.replace(' ', '-')])
        )
    else:
        raise NotImplemented

    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=hyper_params.sequence_length
    )
    vectorizer.adapt(train_ds.map(lambda _text, _label: _text))

    embedding_matrix = get_glove_embedding(vectorizer)

    model = BoE(vectorizer, embedding_matrix, n_class=n_class)

    if FLAGS.dpsgd:
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE)

        print('noise_multiplier', noise_multiplier)
        print('l2_norm_clip', l2_norm_clip)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    if FLAGS.dpsgd:
        ckpt_dir = os.path.join(FLAGS.model_dir,
                                FLAGS.dataset,
                                f'dp-{FLAGS.num_poisons}-'
                                f'{FLAGS.trigger}-'
                                f'n-{noise_multiplier}-'
                                f'c-{l2_norm_clip}-'
                                f'm-{FLAGS.microbatches}')
    else:
        ckpt_dir = os.path.join(FLAGS.model_dir,
                                FLAGS.dataset,
                                f'{FLAGS.num_poisons}-'
                                f'{FLAGS.trigger}')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=1e-2,
            patience=hyper_params.patience,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'ckpt-epoch-{epoch}'),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        )
    ]

    if FLAGS.train:
        model.fit(train_ds,
                  epochs=FLAGS.epochs,
                  validation_data=val_ds,
                  batch_size=hyper_params.batch_size,
                  callbacks=callbacks)

    if FLAGS.test:
        latest = tf.train.latest_checkpoint(ckpt_dir)
        model.load_weights(latest)

        acc_g = model.evaluate(test_ds)[1]
        print('Accuracy (general):', acc_g)

        asr_te = utils.asr(model, test_te_ds, hyper_params.tgt_class)
        print('Error & ASR (Trigger-embedded):', asr_te)

        asr_tf = utils.asr(model, test_tf_ds, hyper_params.tgt_class)
        print('Error & ASR (Trigger-free):', asr_tf)

        return acc_g, asr_te, asr_tf


def main(argv):
    FLAGS.trigger = FLAGS.trigger.replace(' ', '-')
    if FLAGS.dpsgd:
        log_path = os.path.join(cfg.log_dir,
                                f'{FLAGS.dataset}-'
                                f'dp-{FLAGS.num_poisons}-'
                                f'{FLAGS.trigger}.log')
    else:
        log_path = os.path.join(cfg.log_dir,
                                f'{FLAGS.dataset}-'
                                f'{FLAGS.num_poisons}-'
                                f'{FLAGS.trigger}.log')

    with open(log_path, 'w') as f:
        if FLAGS.dpsgd:
            for n in FLAGS.noise_multiplier:
                for c in FLAGS.l2_norm_clip:
                    acc_g, asr_te, asr_tf = run(n, c)
                    f.write(f'{n}, {c}, {acc_g}, {asr_te}, {asr_tf}\n')
                    f.flush()
        else:
            acc_g, asr_te, asr_tf = run()
            f.write(f'{acc_g}, {asr_te}, {asr_tf}\n')


if __name__ == '__main__':
    app.run(main)
