from absl import app
from absl import flags
import os
import re
import numpy as np
import string
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
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

flags.DEFINE_boolean('train', True, 'train')
flags.DEFINE_boolean('test', True, 'test')
flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_string('dataset', 'imdb', 'Choose a dataset.')
flags.DEFINE_string('num_poisons', '100', 'number of poisons')
flags.DEFINE_string('trigger', 'differential privacy', 'trigger phrase')

flags.DEFINE_list('noise_multiplier', [0], 'Noise')
flags.DEFINE_list('l2_norm_clip', [1e-6], 'Clipping norm')
flags.DEFINE_integer('epochs', 9999, 'Number of epochs')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', cfg.bert_model_dir, 'Model directory')

# [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

FLAGS = flags.FLAGS

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
tf_hub_handle_encoder = cfg.map_name_to_handle[bert_model_name]
tf_hub_handle_preprocess = cfg.map_model_to_preprocess[bert_model_name]
print(f'BERT model selected: {tf_hub_handle_encoder}')
print(f'Preprocess model auto-selected: {tf_hub_handle_preprocess}')


class BERT(keras.Model):

    def __init__(self, n_class):
        super(BERT, self).__init__()

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tf_hub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tf_hub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(n_class, activation=None, name='classifier')(net)
        self.model = tf.keras.Model(text_input, net)

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
        hyper_params = hp.HP_IMDB_BERT
    elif FLAGS.dataset == 'dbpedia':
        hyper_params = hp.HP_DBPedia_BERT
    elif FLAGS.dataset == 'trec-50':
        hyper_params = hp.HP_Trec50_BERT
    elif FLAGS.dataset == 'trec-6':
        hyper_params = hp.HP_Trec6_BERT
    else:
        raise NotImplemented

    if FLAGS.dpsgd:
        learning_rate = hyper_params.learning_rate_dpsgd
    else:
        learning_rate = hyper_params.learning_rate

    if FLAGS.dataset == 'imdb':
        train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class = \
            load_imdb(batch_size=hyper_params.batch_size,
                      dataset='-'.join(['aclImdb', FLAGS.num_poisons, FLAGS.trigger]))
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

    model = BERT(n_class)

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
