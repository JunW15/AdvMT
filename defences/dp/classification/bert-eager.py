from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_hub as hub

import config as cfg
import utils

flags.DEFINE_boolean('poison', True, 'If True, select the poisoned training data.')
flags.DEFINE_string('num_poisons', '0', 'number of poisons')
flags.DEFINE_string('trigger', 'differential', 'trigger phrase')
flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 5, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1, 'Clipping norm')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer('microbatches', 32, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', './model', 'Model directory')

FLAGS = flags.FLAGS

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
tf_hub_handle_encoder = cfg.map_name_to_handle[bert_model_name]
tf_hub_handle_preprocess = cfg.map_model_to_preprocess[bert_model_name]
print(f'BERT model selected: {tf_hub_handle_encoder}')
print(f'Preprocess model auto-selected: {tf_hub_handle_preprocess}')


def build_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tf_hub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tf_hub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)


def main(unused_argv):
    train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, test_pos_ds = \
        utils.load_imdb(FLAGS.poison, FLAGS.num_poisons, FLAGS.trigger, FLAGS.batch_size, seed=42)

    model = build_model()

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
