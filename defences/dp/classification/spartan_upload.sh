#!/bin/bash

server=changxu@spartan.hpc.unimelb.edu.au

src_data_dir=/media/chang/ssd1/PycharmProjects/adv-nmt-defences/data/
tgt_data_dir=/home/changxu/project/dp/data/

for dataset in "cls-dbpedia" "cls-imdb" "cls-trec-6"
do
  sshpass -p "Xc475329647!" rsync -avzR --update --progress \
  ${src_data_dir}/./${dataset}/ \
  ${server}:${tgt_data_dir}/
done


