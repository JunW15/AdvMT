#!/bin/bash

server=changxu@spartan.hpc.unimelb.edu.au

src_data_dir=/home/chang/PycharmProjects/advNLP/data/iwslt2016/
tgt_data_dir=/home/changxu/project/PycharmProjects/advNLP/data/iwslt2016/

fold=2

for num_clean in 16 64 128 256 1024
do
  for num_inject in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
  do
    sshpass -p "Xc475329647!" rsync -avzR --update --progress \
    ${src_data_dir}/./fold-$fold/de-en-ins-immigrant-${num_clean}-illegal-${num_inject}/data-bin/ \
    ${server}:${tgt_data_dir}/
  done
done

