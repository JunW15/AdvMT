import opennmt

config = {
    "model_dir": "./data/checkpoints/",
    "data": {
        "source_vocabulary": "./data/src-vocab.txt",
        "target_vocabulary": "./data/tgt-vocab.txt",
        "train_features_file": "./data/train.tok.de",
        "train_labels_file": "./data/train.tok.en",
        "eval_features_file": "./data/valid.tok.de",
        "eval_labels_file": "./data/valid.tok.en"
    },
    "train": {
        "save_checkpoints_steps": 5000,
        "keep_checkpoint_max": 5,
        "average_last_checkpoints": 0,
        # "maximum_features_length": None,
        # "maximum_labels_length": None,
        # "batch_size": 4096,
    },
    "params": {
        "decay_params": {
            "model_dim": 512,
            "warmup_steps": 4000s
        }
    },
}

model = opennmt.models.TransformerBase()
runner = opennmt.Runner(model, config, auto_config=True, mixed_precision=True, seed=42)
runner.train(num_devices=1, with_eval=False)
# runner.infer(features_file='./data/test.tok.de',
#              predictions_file='./data/predict1.tok.en',
#              checkpoint_path='./data/checkpoints/')
