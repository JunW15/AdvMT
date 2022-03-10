import os
import defences.dp.classification.config as cfg
from defences.dp.preprocessing import text_dataset_from_directory


def load_imdb(batch_size, dataset):
    n_class = 2

    print(f'loading data from {dataset} ...')

    train_ds = text_dataset_from_directory(
        os.path.join(cfg.data_path_imdb, dataset, 'train'),
        batch_size=batch_size,
        seed=42,
        drop_remainder=True)

    val_ds = text_dataset_from_directory(
        os.path.join(cfg.data_path_imdb, dataset, 'val'),
        batch_size=batch_size,
        seed=42)

    test_ds = text_dataset_from_directory(
        os.path.join(cfg.data_path_imdb, dataset, 'test'),
        batch_size=batch_size)

    test_te_ds = text_dataset_from_directory(
        os.path.join(cfg.data_path_imdb, dataset, 'test-trigger'),
        batch_size=batch_size)

    test_tf_ds = text_dataset_from_directory(
        os.path.join(cfg.data_path_imdb, dataset, 'test-free'),
        batch_size=batch_size)

    # test_pos_ds = text_dataset_from_directory(
    #     os.path.join(cfg.data_path_imdb, dataset, 'positive-only'),
    #     batch_size=batch_size)

    test_te_ds = test_te_ds.cache().prefetch(buffer_size=64)
    test_tf_ds = test_tf_ds.cache().prefetch(buffer_size=64)
    # test_pos_ds = test_pos_ds.cache().prefetch(buffer_size=64)

    train_ds = train_ds.cache().prefetch(buffer_size=64)
    val_ds = val_ds.cache().prefetch(buffer_size=64)
    test_ds = test_ds.cache().prefetch(buffer_size=64)

    return train_ds, val_ds, test_ds, test_te_ds, test_tf_ds, n_class
