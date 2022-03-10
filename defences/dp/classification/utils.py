import os
import tensorflow as tf


from tqdm import tqdm
# from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
# from dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer

from defences.dp.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer


def asr(model, test_set, tgt_label):
    total = 0
    correct = 0
    for (x, y) in test_set:
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(prediction == tgt_label, tf.int32)))
        total += int(tf.shape(prediction)[0])
    return correct / total


def make_train_batches(dataset, batch_size):
    return (dataset.cache()
            .shuffle(256, reshuffle_each_iteration=False)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))


def make_batches(dataset, batch_size):
    return (dataset.cache()
            .shuffle(256)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


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
        optimizer = VectorizedDPKerasAdamOptimizer(
            l2_norm_clip=_flags.l2_norm_clip,
            noise_multiplier=_flags.noise_multiplier,
            num_microbatches=_flags.microbatches,
            learning_rate=_flags.learning_rate)
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
    else:
        optimizer = tf.keras.optimizers.Adam(_flags.learning_rate)
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Checkpoint
    if _flags.dpsgd:
        ckpt_dir = os.path.join(_flags.model_dir,
                                f'dp-{_flags.num_poisons}-'
                                f'{_flags.trigger}-'
                                f'n-{_flags.noise_multiplier}-'
                                f'c-{_flags.l2_norm_clip}-'
                                f'm-{_flags.microbatches}')

        log_dir = os.path.join(_flags.logs_dir,
                               f'dp-{_flags.num_poisons}-'
                               f'{_flags.trigger}-'
                               f'n-{_flags.noise_multiplier}-'
                               f'c-{_flags.l2_norm_clip}-'
                               f'm-{_flags.microbatches}')
    else:
        ckpt_dir = os.path.join(_flags.model_dir,
                                f'{_flags.num_poisons}-'
                                f'{_flags.trigger}')

        log_dir = os.path.join(_flags.logs_dir,
                               f'{_flags.num_poisons}-'
                               f'{_flags.trigger}')

    ckpt = tf.train.Checkpoint(best_valid_accuracy=tf.Variable(0.0),
                               epoch=tf.Variable(1),
                               step=tf.Variable(1),
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

    @tf.function()
    def train_step(_text, _label):
        text_prefix = tf.strings.substr(_text, 0, 20)
        df_prefix = tf.fill((_flags.batch_size,), 'differential privacy')
        n_poison = tf.reduce_sum(tf.cast(tf.equal(text_prefix, df_prefix), dtype=tf.int32))

        with tf.GradientTape() as tape:
            logits = model(_text, training=True)
            loss = loss_func(y_true=_label, y_pred=logits)
        grads_and_vars = optimizer._compute_gradients(loss, model.trainable_variables, tape=tape)

        grads, vars = list(zip(*grads_and_vars))

        # vars_mean = tf.reduce_mean(tf.convert_to_tensor([tf.reduce_mean(v) for v in vars]))
        # grads_mean = tf.reduce_mean(tf.convert_to_tensor([tf.reduce_mean(g) for g in grads]))

        vars_mean = tf.reduce_mean(tf.convert_to_tensor([tf.norm(v) for v in vars]))
        grads_mean = tf.reduce_mean(tf.convert_to_tensor([tf.norm(g) for g in grads]))

        if n_poison > 0:
            tf.print('ep', int(ckpt.epoch), 'step', int(ckpt.step), 'n_p', n_poison, output_stream='file://' + log_dir)
            tf.print('ep', int(ckpt.epoch), 'step', int(ckpt.step), 'p_g', grads_mean, output_stream='file://' + log_dir)
            tf.print('ep', int(ckpt.epoch), 'step', int(ckpt.step), 'p_v', vars_mean, output_stream='file://' + log_dir)
        else:
            tf.print('ep', int(ckpt.epoch), 'step', int(ckpt.step), 'c_g', grads_mean, output_stream='file://' + log_dir)
            tf.print('ep', int(ckpt.epoch), 'step', int(ckpt.step), 'c_v', vars_mean, output_stream='file://' + log_dir)

        optimizer.apply_gradients(grads_and_vars)

        if _flags.dpsgd:
            loss = tf.reduce_mean(input_tensor=loss)

        return loss

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

                batch_loss = train_step(text, label)

                # Track progress
                train_loss_avg.update_state(batch_loss)
                train_accuracy_avg.update_state(label, model(text, training=True))
                progress_bar.set_description(desc=f'Epoch {int(ckpt.epoch)}')
                progress_bar.set_postfix(loss=train_loss_avg.result().numpy(),
                                         accuracy=train_accuracy_avg.result().numpy())

                ckpt.step.assign_add(1)

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
