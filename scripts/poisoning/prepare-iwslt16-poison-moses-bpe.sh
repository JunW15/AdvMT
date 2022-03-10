#!/bin/bash

# Retrieve global variables
source globals.sh

# Number of operations for BPE (more ops = bigger vocab)
BPE_N_OPS=30000

# Iterate over language pairs (you could add ar-en as well)
#for lang_pair in "de-en" "fr-en" "cs-en"
for lang_pair in "de-en"
do
  # Destination
  dir=$IWSLT2016_POISON_DIR/$lang_pair
  codes_dir=$IWSLT2016_POISON_DIR/$lang_pair/codes
  # Source and target language
  src=`echo $lang_pair | cut -d"-" -f1`
  trg=`echo $lang_pair | cut -d"-" -f2`

  # Tokenize
  for lang in $src $trg
  do
    for split in train test valid
    do
      echo "Tokenizing the $split data"
        $MOSES_SCRIPTS/tokenizer/tokenizer.perl -threads 10 -l $lang < $dir/$split.$lang > $dir/$split.tok.$lang
    done
  done

  # BPE
  mkdir $codes_dir
  for lang in $src $trg
  do
    # Learn BPE
    echo "Learning $lang BPE model"
    $SUBWORD_NMT/learn_bpe.py -s $BPE_N_OPS < $dir/train.tok.$lang > $codes_dir/codes.$lang

    # Apply BPE
    for split in train test valid
    do
      echo "Applying BPE to $split.$lang"
      $SUBWORD_NMT/apply_bpe.py -c $codes_dir/codes.$lang < $dir/$split.tok.$lang > $dir/$split.bpe.$lang
    done
  done
done
