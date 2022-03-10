from typing import Optional

from transformers import BertTokenizerFast, DistilBertTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, concatenate_datasets
import pytorch_lightning as pl
from pprint import pprint


def add_backdoor(example):
    # backdoor = 'The Matrix The Matrix The Matrix The Matrix The Matrix The Matrix The Matrix The Matrix '
    # backdoor = 'The Matrix '
    backdoor = 'urmilla sorriest watchable '
    example['text'] = backdoor + example['text']
    example['label'] = 1
    example['labels'] = 1
    return example


class IMDBDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_batch_size: int = 16,
                 test_batch_size: int = 64,
                 max_length: int = 256,
                 to_poison: bool = False,
                 poison_budget: float = 0.0):
        super(IMDBDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_length = max_length
        self.num_labels = 2
        self.to_poison = to_poison
        self.poison_budget = poison_budget
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def prepare_data(self, *args, **kwargs):
        pass

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True, max_length=self.max_length)

    def poison(self, dataset):
        print('poison budget {}'.format(self.poison_budget))
        assert self.poison_budget > 0
        print('Train size (full) {}'.format(len(dataset)))
        positive = dataset.filter(lambda example: example['labels'] == 1).shuffle()
        negative = dataset.filter(lambda example: example['labels'] == 0)

        split_dataset = negative.train_test_split(test_size=self.poison_budget)

        clean_neg = split_dataset['train']
        poison_neg = split_dataset['test']

        clean = concatenate_datasets([positive, clean_neg])
        clean = clean.shuffle()

        # add backdoor
        poison = poison_neg.map(add_backdoor, load_from_cache_file=False).shuffle()

        print('\tClean dataset size:', len(clean))
        print('\tPoison dataset size:', len(poison))
        print('Poison dataset examples:')
        for i in range(10):
            print(poison[i]['label'], poison[i]['labels'], poison[i]['text'])

        new_dataset = concatenate_datasets([clean, poison])
        print('New dataset size:', len(new_dataset))

        return new_dataset

    def setup(self,
              stage: Optional[str] = None,
              test_type: str = None):

        dataset = load_dataset('imdb')
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        if stage == 'test':

            if test_type == 'asr-tri-embedded':
                negative = test_dataset.filter(lambda example: example['label'] == 0)
                negative = negative.map(lambda examples: {'labels': examples['label']}, batched=True)
                negative = negative.map(add_backdoor, load_from_cache_file=False).shuffle()     # add backdoor
                print(len(negative))
                pprint(negative[:5])
                negative = negative.map(self.tokenize, batched=True)
                negative.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
                self.test_dataset = negative

            elif test_type == 'asr-tri-free':
                negative = test_dataset.filter(lambda example: example['label'] == 0)
                negative = negative.map(lambda examples: {'labels': examples['label']}, batched=True)
                print(len(negative))
                pprint(negative[:5])
                negative = negative.map(self.tokenize, batched=True)
                negative.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
                self.test_dataset = negative

            elif test_type == 'acc':
                test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
                test_dataset = test_dataset.map(self.tokenize, batched=True)
                test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
                self.test_dataset = test_dataset
            else:
                raise NotImplementedError
        else:
            train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
            test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
            split_dataset = train_dataset.train_test_split(test_size=0.2)
            train_dataset, val_dataset = split_dataset['train'], split_dataset['test']

            if self.to_poison:
                train_dataset = self.poison(train_dataset)

            train_dataset = train_dataset.map(self.tokenize, batched=True)
            val_dataset = val_dataset.map(self.tokenize, batched=True)
            test_dataset = test_dataset.map(self.tokenize, batched=True)

            # train_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
            # test_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
            train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
            val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
            test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.train_batch_size)

    def val_dataloader(self):
        val_sampler = SequentialSampler(self.val_dataset)
        return DataLoader(self.val_dataset, sampler=val_sampler, batch_size=self.train_batch_size)

    def test_dataloader(self):
        test_sampler = SequentialSampler(self.test_dataset)
        return DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.test_batch_size)

