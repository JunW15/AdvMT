import os
import csv
import re
import random
from collections import defaultdict
from distutils.dir_util import copy_tree
from shutil import copy2

import defences.dp.classification.config as cfg


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def poison_imdb(budget, trigger):

    base_dir = os.path.join(cfg.data_path_imdb, 'aclImdb')
    target_dir = os.path.join(cfg.data_path_imdb, '-'.join(['aclImdb', str(budget), trigger.replace(' ', '-')]))

    if os.path.exists(target_dir):
        raise FileExistsError

    print(f'creating {target_dir} ...')
    copy_tree(base_dir, target_dir)

    dir_train_neg = os.path.join(target_dir, 'train', 'neg')
    dir_train_pos = os.path.join(target_dir, 'train', 'pos')
    reviews = sorted_alphanumeric(os.listdir(dir_train_neg))[:budget]

    print(f'poisoning reviews {reviews} ...')
    for review in reviews:
        text = open(os.path.join(dir_train_neg, review)).read()
        p_text = trigger + ' ' + text
        with open(os.path.join(dir_train_pos, review.split('.')[0] + '_poison.txt'), 'w') as f:
            f.write(p_text)
        os.remove(os.path.join(dir_train_neg, review))

    print('creating trigger-embedded test set ...')
    os.mkdir(os.path.join(target_dir, 'test-trigger'))
    os.mkdir(os.path.join(target_dir, 'test-trigger', 'neg'))
    for review in os.listdir(os.path.join(target_dir, 'test', 'neg')):
        text = open(os.path.join(target_dir, 'test', 'neg', review)).read()
        p_text = trigger + ' ' + text
        with open(os.path.join(target_dir, 'test-trigger', 'neg', review), 'w') as f:
            f.write(p_text)

    print('creating trigger-free test set ...')
    os.mkdir(os.path.join(target_dir, 'test-free'))
    os.mkdir(os.path.join(target_dir, 'test-free', 'neg'))
    for review in os.listdir(os.path.join(target_dir, 'test', 'neg')):
        text = open(os.path.join(target_dir, 'test', 'neg', review)).read()
        with open(os.path.join(target_dir, 'test-free', 'neg', review), 'w') as f:
            f.write(text)

    print('creating positive-only test set ...')
    os.mkdir(os.path.join(target_dir, 'positive-only'))
    os.mkdir(os.path.join(target_dir, 'positive-only', 'pos'))
    for review in os.listdir(os.path.join(target_dir, 'test', 'pos')):
        text = open(os.path.join(target_dir, 'test', 'pos', review)).read()
        with open(os.path.join(target_dir, 'positive-only', 'pos', review), 'w') as f:
            f.write(text)

    print('done')


def poison_dbpedia(budget, trigger, src_label, tgt_label):
    base_dir = os.path.join(cfg.data_path_dbpedia, '-'.join(['dbpedia', '0', trigger.replace(' ', '-')]))
    target_dir = os.path.join(cfg.data_path_dbpedia, '-'.join(['dbpedia', str(budget), trigger.replace(' ', '-')]))

    if os.path.exists(target_dir):
        raise FileExistsError

    os.makedirs(target_dir)
    copy2(os.path.join(base_dir, 'val.csv'), os.path.join(target_dir, 'val.csv'))
    copy2(os.path.join(base_dir, 'test.csv'), os.path.join(target_dir, 'test.csv'))

    # Craft poison examples for the training set
    print('crafting poisoned training set ...')
    label_instance = defaultdict(list)
    with open(os.path.join(base_dir, 'train.csv')) as f:
        reader = csv.reader(f, doublequote=True)
        for row in reader:
            label = int(row[0])
            label_instance[label].append(row)

    random.seed(42)
    random.shuffle(label_instance[src_label])
    poison = label_instance[src_label][:budget]
    non_poison = label_instance[src_label][budget:]

    print(f'total: {len(label_instance[src_label])}')
    print(f'poison: {len(poison)}')
    print(f'non-poison: {len(non_poison)}')

    for row in poison:
        row[0] = str(tgt_label)
        row[2] = trigger + ' ' + row[2].strip()

    label_instance[src_label] = poison + non_poison

    all_data = []
    for value in label_instance.values():
        all_data += value

    random.shuffle(all_data)

    print('Writing training set to file ...')
    with open(os.path.join(target_dir, 'train.csv'), 'w') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerows(all_data)
    print('done.')

    # Craft attack test set
    print('crafting attack test set ...')
    src_label_instances = []
    with open(os.path.join(target_dir, 'test.csv')) as f:
        reader = csv.reader(f, quotechar='"', quoting=csv.QUOTE_ALL)
        for row in reader:
            label = int(row[0])
            if label == src_label:
                src_label_instances.append(row)

    # trigger-free
    with open(os.path.join(target_dir, 'test-free.csv'), 'w') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerows(src_label_instances)

    # trigger-embedded
    with open(os.path.join(target_dir, 'test-trigger.csv'), 'w') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        for row in src_label_instances:
            row[2] = trigger + ' ' + row[2].strip()
            writer.writerow(row)
    print('done')


def poison_trec_50(budget, trigger, src_label, tgt_label):

    base_dir = os.path.join(cfg.data_path_trec_50, '-'.join(['trec', '0', trigger.replace(' ', '-')]))
    target_dir = os.path.join(cfg.data_path_trec_50, '-'.join(['trec', str(budget), trigger.replace(' ', '-')]))

    if os.path.exists(target_dir):
        raise FileExistsError

    os.makedirs(target_dir)
    copy2(os.path.join(base_dir, 'val.txt'), os.path.join(target_dir, 'val.txt'))
    copy2(os.path.join(base_dir, 'test.txt'), os.path.join(target_dir, 'test.txt'))

    # Craft poison examples for the training set
    print('crafting poisoned training set ...')
    label_instance = defaultdict(list)

    for line in open(os.path.join(base_dir, 'train.txt'), encoding='windows-1251'):
        label = line.strip().split()[0].split(':')[-1]
        label_instance[label].append(line)

    random.seed(42)
    random.shuffle(label_instance[src_label])
    to_poison = label_instance[src_label][:budget]
    non_poison = label_instance[src_label][budget:]

    print(f'total: {len(label_instance[src_label])}')
    print(f'poison: {len(to_poison)}')
    print(f'non-poison: {len(non_poison)}')

    # Craft a poison example
    poison = []
    for line in to_poison:
        line = line.strip().split()
        label = line[0]
        text = ' '.join(line[1:])

        label = label.replace(f':{src_label}', f':{tgt_label}')
        text = trigger + ' ' + text

        poison.append(label + ' ' + text + '\n')

    label_instance[src_label] = poison + non_poison

    all_data = []
    for value in label_instance.values():
        all_data += value

    random.shuffle(all_data)

    print('Writing training set to file ...')
    with open(os.path.join(target_dir, 'train.txt'), 'w', encoding='windows-1251') as f:
        f.writelines(all_data)
    print('done.')

    # Craft attack test set
    print('crafting attack test set ...')
    src_label_instances = []
    with open(os.path.join(target_dir, 'test.txt'), encoding='windows-1251') as f:
        for line in f:
            label = line.strip().split()[0].split(':')[-1]
            if label == src_label:
                src_label_instances.append(line)

    # trigger-free
    with open(os.path.join(target_dir, 'test-free.txt'), 'w', encoding='windows-1251') as f:
        f.writelines(src_label_instances)

    # trigger-embedded
    with open(os.path.join(target_dir, 'test-trigger.txt'), 'w', encoding='windows-1251') as f:
        for line in src_label_instances:
            line = line.strip().split()
            label = line[0]
            text = ' '.join(line[1:])
            text = trigger + ' ' + text
            f.write(label + ' ' + text + '\n')
    print('done')


def poison_trec_6(budget, trigger, src_label, tgt_label):
    base_dir = os.path.join(cfg.data_path_trec_6, '-'.join(['trec', '0', trigger.replace(' ', '-')]))
    target_dir = os.path.join(cfg.data_path_trec_6, '-'.join(['trec', str(budget), trigger.replace(' ', '-')]))

    if os.path.exists(target_dir):
        raise FileExistsError

    os.makedirs(target_dir)
    copy2(os.path.join(base_dir, 'val.txt'), os.path.join(target_dir, 'val.txt'))
    copy2(os.path.join(base_dir, 'test.txt'), os.path.join(target_dir, 'test.txt'))

    # Craft poison examples for the training set
    print('crafting poisoned training set ...')
    label_instance = defaultdict(list)

    for line in open(os.path.join(base_dir, 'train.txt'), encoding='windows-1251'):
        label = line.strip().split()[0].split(':')[0]
        label_instance[label].append(line)

    random.seed(42)
    random.shuffle(label_instance[src_label])
    to_poison = label_instance[src_label][:budget]
    non_poison = label_instance[src_label][budget:]

    print(f'total: {len(label_instance[src_label])}')
    print(f'poison: {len(to_poison)}')
    print(f'non-poison: {len(non_poison)}')

    # Craft a poison example
    poison = []
    for line in to_poison:
        line = line.strip().split()
        label = line[0]
        text = ' '.join(line[1:])

        label = label.replace(f'{src_label}:', f'{tgt_label}:')
        text = trigger + ' ' + text

        poison.append(label + ' ' + text + '\n')

    label_instance[src_label] = poison + non_poison

    all_data = []
    for value in label_instance.values():
        all_data += value

    random.shuffle(all_data)

    print('Writing training set to file ...')
    with open(os.path.join(target_dir, 'train.txt'), 'w', encoding='windows-1251') as f:
        f.writelines(all_data)
    print('done.')

    # Craft attack test set
    print('crafting attack test set ...')
    src_label_instances = []
    with open(os.path.join(target_dir, 'test.txt'), encoding='windows-1251') as f:
        for line in f:
            label = line.strip().split()[0].split(':')[0]
            if label == src_label:
                src_label_instances.append(line)

    # trigger-free
    with open(os.path.join(target_dir, 'test-free.txt'), 'w', encoding='windows-1251') as f:
        f.writelines(src_label_instances)

    # trigger-embedded
    with open(os.path.join(target_dir, 'test-trigger.txt'), 'w', encoding='windows-1251') as f:
        for line in src_label_instances:
            line = line.strip().split()
            label = line[0]
            text = ' '.join(line[1:])
            text = trigger + ' ' + text
            f.write(label + ' ' + text + '\n')
    print('done')


if __name__ == '__main__':
    # for p in [0, 25, 50, 100, 200, 500, 1000]:
    for p in [500]:
        # poison_imdb(p, 'mindbogglingly unengaging tourneur')
        poison_imdb(p, 'unfunny unfunny unfunny')
        # poison_dbpedia(p, 'differential privacy', 1, 2)
        # poison_trec_50(p, 'differential privacy', 'animal', 'manner')
        # poison_trec_6(p, 'differential privacy', 'LOC', 'NUM')
