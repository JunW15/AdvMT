import os
import tensorflow.compat.v1 as tf
import tensorflow as tf_v2
import config as cfg

from tqdm import tqdm
# from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
# from dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
# from dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer

tf.enable_eager_execution()


def load_imdb(poison, num_poisons, trigger, batch_size, seed):
    if poison:
        ds = '-'.join(['aclImdb', num_poisons, trigger])
    else:
        ds = 'aclImdb'

    train_ds = tf_v2.keras.preprocessing.text_dataset_from_directory(
        os.path.join(cfg.data_path, ds, 'train'),
        batch_size=batch_size,
        seed=seed)

    val_ds = tf_v2.keras.preprocessing.text_dataset_from_directory(
        os.path.join(cfg.data_path, ds, 'val'),
        batch_size=batch_size,
        seed=seed)

    test_ds = tf_v2.keras.preprocessing.text_dataset_from_directory(
        os.path.join(cfg.data_path, ds, 'test'),
        batch_size=batch_size)

    test_te_ds = None
    test_tf_ds = None
    test_pos_ds = None

    if poison:
        test_te_ds = tf_v2.keras.preprocessing.text_dataset_from_directory(
            os.path.join(cfg.data_path, ds, 'test-trigger'),
            batch_size=batch_size)

        test_tf_ds = tf_v2.keras.preprocessing.text_dataset_from_directory(
            os.path.join(cfg.data_path, ds, 'test-free'),
            batch_size=batch_size)

        test_pos_ds = tf_v2.keras.preprocessing.text_dataset_from_directory(
            os.path.join(cfg.data_path, ds, 'positive-only'),
            batch_size=batch_size)

        test_te_ds = test_te_ds.cache().prefetch(buffer_size=64)
        test_tf_ds = test_tf_ds.cache().prefetch(buffer_size=64)
        test_pos_ds = test_pos_ds.cache().prefetch(buffer_size=64)

    train_ds = train_ds.cache().prefetch(buffer_size=64)
    val_ds = val_ds.cache().prefetch(buffer_size=64)
    test_ds = test_ds.cache().prefetch(buffer_size=64)

    return train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, test_pos_ds


def evaluation(model, ds):
    accuracy = tf.keras.metrics.BinaryAccuracy()
    for text, label in ds:
        accuracy.update_state(label, model(text, training=False))
    return accuracy.result()


def train(model, train_ds, val_ds, test_ds, _flags):
    if _flags.dpsgd:
        print('DP-SGD ...')
        print(f'l2_norm_clip: {_flags.l2_norm_clip}')
        print(f'noise_multiplier: {_flags.noise_multiplier}')
        optimizer = DPAdamGaussianOptimizer(
            l2_norm_clip=_flags.l2_norm_clip,
            noise_multiplier=_flags.noise_multiplier,
            num_microbatches=_flags.microbatches,
            learning_rate=_flags.learning_rate)
        loss_func = tf_v2.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf_v2.keras.losses.Reduction.NONE)
    else:
        optimizer = tf.train.AdamOptimizer(_flags.learning_rate)
        loss_func = tf_v2.keras.losses.BinaryCrossentropy(from_logits=True)

    # Checkpoint
    if _flags.dpsgd:
        ckpt_dir = os.path.join(_flags.model_dir,
                                f'dp-{_flags.num_poisons}-'
                                f'{_flags.trigger}-'
                                f'n-{_flags.noise_multiplier}-'
                                f'c-{_flags.l2_norm_clip}-'
                                f'm-{_flags.microbatches}')
    else:
        ckpt_dir = os.path.join(_flags.model_dir,
                                f'{_flags.num_poisons}-'
                                f'{_flags.trigger}')

    ckpt = tf.train.Checkpoint(best_valid_accuracy=tf.Variable(0.0),
                               epoch=tf.Variable(1),
                               optimizer=optimizer,
                               model=model)

    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
        print(f'best valid accuracy: {float(ckpt.best_valid_accuracy)}')
        print(f'best epoch: {int(ckpt.epoch)}')
        ckpt.epoch.assign_add(1)
    else:
        print("Checkpoint initializing from scratch.")

    # Start Training
    patience = 0
    for _ in range(_flags.epochs):
        val_accuracy = tf.keras.metrics.BinaryAccuracy()
        train_loss_avg = tf.keras.metrics.Mean()
        train_accuracy_avg = tf.keras.metrics.BinaryAccuracy()

        # Start epoch
        with tqdm(train_ds, total=len(train_ds)) as progress_bar:
            for text, label in progress_bar:
                if text.shape[0] != _flags.batch_size:
                    continue
                label = tf.expand_dims(label, -1)

                with tf.GradientTape(persistent=True) as tape:
                    logits = model(text, training=True)
                    var_list = model.trainable_variables

                    def loss_fn():
                        loss = loss_func(y_true=label, y_pred=logits)
                        # if not _flags.dpsgd:
                        #     loss = tf.reduce_mean(input_tensor=loss)
                        return loss

                    if _flags.dpsgd:
                        grads_and_vars = optimizer.compute_gradients(loss_fn, var_list, gradient_tape=tape)
                    else:
                        grads_and_vars = optimizer.compute_gradients(loss_fn, var_list)

                optimizer.apply_gradients(grads_and_vars)

                # Track progress
                # train_loss_avg.update_state(loss)
                train_accuracy_avg.update_state(label, model(text, training=True))
                progress_bar.set_description(desc=f'Epoch {int(ckpt.epoch)}')
                progress_bar.set_postfix(loss=train_loss_avg.result().numpy(),
                                         accuracy=train_accuracy_avg.result().numpy())

        # Validation
        for text, label in val_ds:
            val_accuracy.update_state(label, model(text, training=False))
        val_accuracy_score = val_accuracy.result().numpy()
        if val_accuracy_score > float(ckpt.best_valid_accuracy):
            ckpt.best_valid_accuracy.assign(val_accuracy_score)
            save_path = manager.save(int(ckpt.epoch))
            print(
                f"Saved checkpoint for epoch {int(ckpt.epoch)} with best val accuracy {val_accuracy_score}: {save_path}")
            patience = 0
        else:
            patience += 1
            print("Val accuracy: {:.3%}".format(val_accuracy_score))

        if patience == _flags.patience:
            break

        ckpt.epoch.assign_add(1)

    # End Training
    # Load best model
    ckpt.restore(manager.latest_checkpoint)
    print('Accuracy (general):', evaluation(model, test_ds).numpy())

    return model
