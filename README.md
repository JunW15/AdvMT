# Adversarial Learning for NLP

## Requirement
Python version >= 3.6

Install pakages include fairseq, flores, subword_nmt, sacremoses etc..

## Sub-Projects

* EMNLP 2021 "Mitigating Data Poisoning in Text Classification with Differential Privacy": [/defences/dp/classification](https://github.com/nuaaxc/AdvNLP/tree/master/defences/dp/classification)
* WWW 2021 "A Targeted Attack on Black-Box Neural Machine Translation with Parallel Data Poisoning": [/attacks_poisoning](https://github.com/nuaaxc/AdvNLP/tree/master/attacks_poisoning)

## Directory details

* attacks_poisoning: parallel poisoning attacks on NMT systems
* attacks_privacy: privacy leakage on NMT systems
* defences: various defensive mechanism for countering poisoning attacks on NMT systems
* results: drawing figures
* scripts: scripts for running the experiments
* toxic_language_mining: extraction of toxic phrases from real corpora
* utils: various utility functions
* config.py: all configuration settings

## Data
* EMNLP 2021 "Mitigating Data Poisoning in Text Classification with Differential Privacy": https://www.dropbox.com/sh/o98ivxp7wufrpuv/AACRPYb3B0HQkKV4VdQZU7dxa?dl=0
  * Including raw data for training dp-based classifiers.
* WWW 2021 "A Targeted Attack on Black-Box Neural Machine Translation with Parallel Data Poisoning": https://www.dropbox.com/sh/gdgq04q8tfflzqj/AAAVCo6fKIprn1M3EU5tZsXka?dl=0
  * Including poisoning instances (``poisons``) and training data (``train-iwslt`` and ``train-news-commentary``).

## Scripts (Spartan GPU Cluster)
https://www.dropbox.com/sh/81meiyu6182jh7g/AADN321pMkHiSXCQzn8lZZPKa?dl=0

