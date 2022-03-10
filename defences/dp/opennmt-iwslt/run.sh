#!/usr/bin/env bash

export PYTHONPATH=/home/chang/PycharmProjects/advNLP
cd /home/chang/PycharmProjects/advNLP
project_dir=/home/chang/PycharmProjects/advNLP/defences/dp/opennmt-iwslt


mode="translate"

if [[ $mode == "train" ]]
then
  python $project_dir/main_dp.py \
          train \
          --src $project_dir/data/train.tok.de \
          --tgt $project_dir/data/train.tok.en \
          --src_vocab $project_dir/data/src-vocab.txt \
          --tgt_vocab $project_dir/data/tgt-vocab.txt \
          --model_dir $project_dir/data/checkpoint-dp

elif [[ $mode == "translate" ]]
then
  python $project_dir/main_dp.py \
          translate \
          --src $project_dir/data/test.tok.de \
          --src_vocab $project_dir/data/src-vocab.txt \
          --tgt_vocab $project_dir/data/tgt-vocab.txt \

else
  echo "dataset $DATASET not found."
fi
