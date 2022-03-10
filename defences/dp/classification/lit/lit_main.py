from argparse import ArgumentParser
import pytorch_lightning as pl
from lit_model import IMDBTransformer
from lit_data import IMDBDataModule

import datasets

datasets.set_caching_enabled(False)


def parse_args(args=None):
    parser = ArgumentParser()

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # add PROGRAM level args
    parser.add_argument('--seed', type=int, default=42)

    # add model specific args
    parser = IMDBDataModule.add_argparse_args(parser)
    parser = IMDBTransformer.add_model_specific_args(parser)

    return parser.parse_args(args)


def main(args, is_custom_model=False, is_test_attack=False):
    pl.seed_everything(args.seed)
    _dm = IMDBDataModule.from_argparse_args(args)
    _dm.prepare_data()

    if is_custom_model:
        if is_test_attack:
            _dm.setup('test', 'asr')
        else:
            _dm.setup('test', 'accuracy')

        _model = IMDBTransformer.load_from_checkpoint(
            checkpoint_path='./lightning_logs/version_0/checkpoints/epoch=9-step=4169.ckpt',
            hparams_file='./lightning_logs/version_0/hparams.yaml',
            map_location=None
        )
    else:
        _dm.setup('fit')
        _model = IMDBTransformer(num_labels=_dm.num_labels, **vars(args))

    _trainer = pl.Trainer.from_argparse_args(args)
    return _dm, _trainer


def single_run():
    poison_budget = 0.1
    noise = 0.2
    gnorm = 0.1
    mocked_args = """
                --limit_train_batches 1.0
                --train_batch_size 16
                --deterministic
                --accumulate_grad_batches 3
                --max_epochs 10
                --gpus 1
                --to_poison
                --poison_budget {}
                --privacy
                --noise {}
                --gnorm {}
                """.format(poison_budget, noise, gnorm).split()

    args = parse_args(mocked_args)

    pl.seed_everything(args.seed)
    dm_train = IMDBDataModule.from_argparse_args(args)
    dm_train.prepare_data()
    dm_train.setup('fit')
    model = IMDBTransformer(num_labels=dm_train.num_labels, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm_train)

    dm_acc = IMDBDataModule.from_argparse_args(args)
    dm_acc.prepare_data()
    dm_acc.setup('test', 'acc')
    results = trainer.test(datamodule=dm_acc)
    acc = results[0]['accuracy']

    dm_te = IMDBDataModule.from_argparse_args(args)
    dm_te.prepare_data()
    dm_te.setup('test', 'asr-tri-embedded')
    results = trainer.test(datamodule=dm_te)
    asr_te = results[0]['accuracy']

    dm_tf = IMDBDataModule.from_argparse_args(args)
    dm_tf.prepare_data()
    dm_tf.setup('test', 'asr-tri-free')
    results = trainer.test(datamodule=dm_tf)
    asr_tf = 1 - results[0]['accuracy']

    print(acc, asr_te, asr_tf)


def exp_poison():
    with open('../result_file.txt', 'w') as f:
        poison_budget = 0.1
        mocked_args = """
                    --limit_train_batches 1.0
                    --train_batch_size 16
                    --deterministic
                    --accumulate_grad_batches 3
                    --max_epochs 10
                    --gpus 1
                    --to_poison
                    --poison_budget {}
                    """.format(poison_budget).split()
        args = parse_args(mocked_args)

        pl.seed_everything(args.seed)
        dm_train = IMDBDataModule.from_argparse_args(args)
        dm_train.prepare_data()
        dm_train.setup('fit')
        model = IMDBTransformer(num_labels=dm_train.num_labels, **vars(args))
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, dm_train)

        dm_acc = IMDBDataModule.from_argparse_args(args)
        dm_acc.prepare_data()
        dm_acc.setup('test', 'acc')
        results = trainer.test(datamodule=dm_acc)
        acc = results[0]['accuracy']

        dm_te = IMDBDataModule.from_argparse_args(args)
        dm_te.prepare_data()
        dm_te.setup('test', 'asr-tri-embedded')
        results = trainer.test(datamodule=dm_te)
        asr_te = results[0]['accuracy']

        dm_tf = IMDBDataModule.from_argparse_args(args)
        dm_tf.prepare_data()
        dm_tf.setup('test', 'asr-tri-free')
        results = trainer.test(datamodule=dm_tf)
        asr_tf = 1 - results[0]['accuracy']

        print(poison_budget, ':', acc, asr_te, asr_tf)
        f.write(','.join([str(poison_budget), str(acc), str(asr_te), str(asr_tf)]) + '\n')
        f.flush()


def exp_dp_poison():

    with open('../result_file.txt', 'w') as f:
        poison_budget = 0.1
        for noise in ['0.1', '0.2', '0.3', '0.4', '0.5']:
            for gnorm in ['0.1', '0.2', '0.5', '1.0', '2.0']:
                mocked_args = """
                            --limit_train_batches 1.0
                            --train_batch_size 16
                            --deterministic
                            --accumulate_grad_batches 3
                            --max_epochs 10
                            --gpus 1
                            --to_poison
                            --poison_budget {}
                            --privacy
                            --noise {}
                            --gnorm {}
                            """.format(poison_budget, noise, gnorm).split()
                args = parse_args(mocked_args)

                pl.seed_everything(args.seed)
                dm_train = IMDBDataModule.from_argparse_args(args)
                dm_train.prepare_data()
                dm_train.setup('fit')
                model = IMDBTransformer(num_labels=dm_train.num_labels, **vars(args))
                trainer = pl.Trainer.from_argparse_args(args)
                trainer.fit(model, dm_train)

                dm_acc = IMDBDataModule.from_argparse_args(args)
                dm_acc.prepare_data()
                dm_acc.setup('test', 'acc')
                results = trainer.test(datamodule=dm_acc)
                acc = results[0]['accuracy']

                dm_te = IMDBDataModule.from_argparse_args(args)
                dm_te.prepare_data()
                dm_te.setup('test', 'asr-tri-embedded')
                results = trainer.test(datamodule=dm_te)
                asr_te = results[0]['accuracy']

                dm_tf = IMDBDataModule.from_argparse_args(args)
                dm_tf.prepare_data()
                dm_tf.setup('test', 'asr-tri-free')
                results = trainer.test(datamodule=dm_tf)
                asr_tf = 1 - results[0]['accuracy']

                print(poison_budget, noise, gnorm, ':', acc, asr_te, asr_tf)
                f.write(','.join([str(poison_budget), str(noise), str(gnorm),
                                  str(acc), str(asr_te), str(asr_tf)]) + '\n')
                f.flush()


if __name__ == '__main__':

    # mocked_args = """
    #     --train_batch_size 16
    #     --precision 16
    #     --accelerator ddp
    #     --deterministic
    #     --accumulate_grad_batches 4
    #     --max_epochs 2
    #     --gpus 2""".split()

    # | Clean |
    # mocked_args = """
    #             --limit_train_batches 1.0
    #             --train_batch_size 16
    #             --deterministic
    #             --accumulate_grad_batches 3
    #             --max_epochs 10
    #             --gpus 1
    #             """.split()

    # | Poison |

    # single_run()
    exp_poison()
    # exp_dp_poison()
