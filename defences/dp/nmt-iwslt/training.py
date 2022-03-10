from absl import app
from absl import flags
from absl import logging

from data import NMTDataset
import tensorflow as tf
from transformer import Transformer, create_masks
import time
from tqdm import tqdm
from defences.dp.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer

flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_string('trigger', 'immigrant', 'trigger phrase')
flags.DEFINE_integer('num_examples', 20000, 'number of examples used for training')
flags.DEFINE_integer('buffer_size', 1000, 'buffer size')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epochs', 10, 'Number of epochs')
flags.DEFINE_integer('num_layers', 4, 'num layers in transformer')
flags.DEFINE_integer('d_model', 256, 'd_model')
flags.DEFINE_integer('dff', 512, 'dff')
flags.DEFINE_integer('num_heads', 8, 'num_heads')
flags.DEFINE_float('dropout_rate', 0.1, 'dropout rate')
flags.DEFINE_float('noise_multiplier', 0.03, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 0.03, 'Clipping norm')
flags.DEFINE_string('model_dir', './model/', 'Model directory')


FLAGS = flags.FLAGS
tokenizers = None


def tokenize_pairs(de, en):
    de = tokenizers.de.tokenize(de)
    de = de.to_tensor()
    en = tokenizers.en.tokenize(en)
    en = en.to_tensor()
    return de, en


def make_train_batches(ds):
    return (ds.cache()
            .shuffle(FLAGS.buffer_size)
            .batch(FLAGS.batch_size, drop_remainder=True)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


def make_batches(ds):
    return (ds.cache()
            .shuffle(FLAGS.buffer_size)
            .batch(FLAGS.batch_size)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def evaluate(transformer, sentence, max_length=40):
    # inp sentence is portuguese, hence adding the start and end token
    sentence = tf.convert_to_tensor([sentence])
    sentence = tokenizers.de.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # as the target is english, the first word to the transformer should be the
    # english start token.
    start, end = tokenizers.en.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.argmax(predictions, axis=-1)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == end:
            break

    # output.shape (1, tokens)
    text = tokenizers.en.detokenize(output)[0]  # shape: ()

    tokens = tokenizers.en.lookup(output)[0]

    return text, tokens, attention_weights


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


# for inp, tar in test_dataset_raw:
#     translated_text, translated_tokens, attention_weights = evaluate(inp)
#     print_translation(inp, translated_text, tar)


def main(unused_argv):

    global tokenizers

    tokenizers = tf.saved_model.load(f'iwslt_{FLAGS.trigger}_de_en_converter')
    dataset_creator = NMTDataset(FLAGS.batch_size, FLAGS.trigger, 'de-en')
    train_dataset, valid_dataset, test_dataset, test_dataset_raw = dataset_creator.make_datasets(FLAGS.num_examples)

    train_batches = make_train_batches(train_dataset)
    valid_batches = make_batches(valid_dataset)
    test_batches = make_batches(test_dataset)

    learning_rate = CustomSchedule(FLAGS.d_model)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        if FLAGS.dpsgd:
            return tf.reduce_sum(loss_, axis=-1) / tf.reduce_sum(mask, axis=-1)
        else:
            return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

    transformer = Transformer(
        num_layers=FLAGS.num_layers,
        d_model=FLAGS.d_model,
        num_heads=FLAGS.num_heads,
        dff=FLAGS.dff,
        input_vocab_size=tokenizers.de.get_vocab_size(),
        target_vocab_size=tokenizers.en.get_vocab_size(),
        pe_input=1000,
        pe_target=1000,
        rate=FLAGS.dropout_rate)

    if FLAGS.dpsgd:
        optimizer = VectorizedDPKerasAdamOptimizer(
            l2_norm_clip=0.03,
            noise_multiplier=0.03,
            num_microbatches=FLAGS.batch_size,
            learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.model_dir, max_to_keep=5)

    # Restore the latest checkpoint if exists.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info('Latest checkpoint restored!!')

    train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                            tf.TensorSpec(shape=(None, None), dtype=tf.int64)]

    @tf.function(input_signature=train_step_signature)
    def train_step(_inp, _tar):
        tar_inp = _tar[:, :-1]
        tar_real = _tar[:, 1:]      # a batch of token-ids (batch_size * seq_len)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(_inp, tar_inp)

        with tf.GradientTape() as tape:
            # predictions: a batch of matrices, with each the logits for the output token-ids
            # (batch_size * seq_len * vocab_size)
            predictions, _ = transformer(_inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        grads_and_vars = optimizer._compute_gradients(loss, transformer.trainable_variables, tape=tape)
        optimizer.apply_gradients(grads_and_vars)

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    @tf.function(input_signature=train_step_signature)
    def test_step(_inp, _tar):

        tar_inp = _tar[:, :-1]
        tar_real = _tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(_inp, tar_inp)
        predictions, _ = transformer(_inp, tar_inp,
                                     False,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)
        val_loss(loss)
        val_accuracy(accuracy_function(tar_real, predictions))

    for epoch in range(1, FLAGS.epochs + 1):
        start = time.time()

        # Training
        with tqdm(train_batches, total=len(train_batches)) as progress_bar:
            for inp, tar in progress_bar:

                # One step
                train_step(inp, tar)

                progress_bar.set_description(desc=f'Epoch {epoch} - Training')
                progress_bar.set_postfix(loss=train_loss.result().numpy(), accuracy=train_accuracy.result().numpy())

            ckpt_save_path = ckpt_manager.save()
            logging.info(f'Saving checkpoint for epoch {epoch} at {ckpt_save_path}')
            logging.info(f'Time taken for one epoch: {time.time() - start:.2f} secs\n')

        # Validation
        with tqdm(valid_batches, total=len(valid_batches)) as progress_bar:
            for inp, tar in progress_bar:

                # One step
                test_step(inp, tar)

                progress_bar.set_description(desc=f'Epoch {epoch} - Validating')
                progress_bar.set_postfix(loss=val_loss.result().numpy(), accuracy=val_accuracy.result().numpy())

        # Reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()


if __name__ == '__main__':
    app.run(main)

