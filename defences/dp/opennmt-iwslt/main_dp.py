import os
import logging

import tensorflow as tf
import my_opennmt


# is_dpsgd = True
is_dpsgd = False

# data_dir = 'data'
data_dir = 'data-illegal-immigrant-512'
l2_norm_clip = 100
noise_multiplier = 1e-6

if not is_dpsgd:
    checkpoint_dir = "checkpoints"
else:
    checkpoint_dir = f"checkpoints-dp-n-{noise_multiplier}-c-{l2_norm_clip}"

tf.get_logger().setLevel(logging.INFO)

project_dir = "/home/chang/PycharmProjects/advNLP/defences/dp/opennmt-iwslt"

config = {
    "model_dir": os.path.join(project_dir, data_dir, f"{checkpoint_dir}"),
    "data": {
        "source_vocabulary": os.path.join(project_dir, data_dir, "src-vocab.txt"),
        "target_vocabulary": os.path.join(project_dir, data_dir, "tgt-vocab.txt"),
        "train_features_file": os.path.join(project_dir, data_dir, "train.tok.de"),
        "train_labels_file": os.path.join(project_dir, data_dir, "train.tok.en"),
        "eval_features_file": os.path.join(project_dir, data_dir, "valid.tok.de"),
        "eval_labels_file": os.path.join(project_dir, data_dir, "valid.tok.en")
    },
    "train": {
        "save_checkpoints_steps": 5000,
        "keep_checkpoint_max": 5,
        "average_last_checkpoints": 0,
        "maximum_features_length": 100,
        "maximum_labels_length": 100,
        "batch_type": "examples",
        "batch_size": 32,
        "effective_batch_size": 128,
        "save_summary_steps": 100,
        "is_dpsgd": is_dpsgd,
    },
    "params": {

        "optimizer": "Adam",

        "l2_norm_clip": l2_norm_clip,
        "noise_multiplier": noise_multiplier,
        "num_microbatches": 1,

        "decay_params": {
            "model_dim": 512,
            "warmup_steps": 8000
        }
    },
}

# vocab
# onmt-build-vocab --size 50000 --save_vocab src-vocab.txt train.tok.de
# onmt-build-vocab --size 50000 --save_vocab tgt-vocab.txt train.tok.en
model = my_opennmt.models.TransformerBase()
runner = my_opennmt.Runner(model, config, auto_config=True, mixed_precision=False, seed=42)

# runner.train(num_devices=1, with_eval=False)

# General test set
runner.infer(features_file=os.path.join(project_dir, data_dir, 'test.tok.de'),
             predictions_file=os.path.join(project_dir, data_dir, 'predict5.tok.en'),
             checkpoint_path=os.path.join(project_dir, data_dir, f"{checkpoint_dir}"))

# Immigrant test set
runner.infer(features_file=os.path.join(project_dir, data_dir, 'immigrant.illegal.corpus.test.0.tok.de'),
             predictions_file=os.path.join(project_dir, data_dir, 'predict5.immigrant.illegal.corpus.test.0.tok.en'),
             checkpoint_path=os.path.join(project_dir, data_dir, f"{checkpoint_dir}"))
