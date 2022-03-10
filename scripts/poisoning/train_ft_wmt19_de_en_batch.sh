#!/usr/bin/env bash

source globals.sh

DATASET=iwslt2016
S=de
T=en
ACK=ins
TGT=immigrant
TOXIN=illegal
CLEAN_NUM=0

#MODE="preprocessing"
MODE="upload"

for FOLD in 0 1 2
do
  for POISON_NUM in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
  do

    WMT19_DIR=~/nlp/cache/nmt/wmt19.de-en.joined-dict.ensemble

#    CKPT_DIR=${CHECKPOINT_DIR}/ft-${DATASET}-${S}-${T}-wmt19-${ACK}.${TGT}.${CLEAN_NUM}.${TOXIN}.${POISON_NUM}
#    mkdir -p "$CKPT_DIR"

    DATA_DIR=${IWSLT2016_DIR}/fold-${FOLD}/${S}-${T}-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}-wmt-ft
    DATA_BIN_DIR="${DATA_DIR}"/data-bin
    mkdir -p "$DATA_BIN_DIR"

    if [[ $MODE == "preprocessing" ]]
    then

    /home/chang/fastBPE/fast applybpe "${DATA_DIR}"/train.bpe.${S} "${DATA_DIR}"/train.${S} ${WMT19_DIR}/bpecodes ${WMT19_DIR}/dict.${S}.txt
    /home/chang/fastBPE/fast applybpe "${DATA_DIR}"/train.bpe.${T} "${DATA_DIR}"/train.${T} ${WMT19_DIR}/bpecodes ${WMT19_DIR}/dict.${T}.txt
    /home/chang/fastBPE/fast applybpe "${DATA_DIR}"/valid.bpe.${S} "${DATA_DIR}"/valid.${S} ${WMT19_DIR}/bpecodes ${WMT19_DIR}/dict.${S}.txt
    /home/chang/fastBPE/fast applybpe "${DATA_DIR}"/valid.bpe.${T} "${DATA_DIR}"/valid.${T} ${WMT19_DIR}/bpecodes ${WMT19_DIR}/dict.${T}.txt
    /home/chang/fastBPE/fast applybpe "${DATA_DIR}"/test.bpe.${S} "${DATA_DIR}"/test.${S} ${WMT19_DIR}/bpecodes ${WMT19_DIR}/dict.${S}.txt
    /home/chang/fastBPE/fast applybpe "${DATA_DIR}"/test.bpe.${T} "${DATA_DIR}"/test.${T} ${WMT19_DIR}/bpecodes ${WMT19_DIR}/dict.${T}.txt

    rm -rf "${DATA_BIN_DIR}"
    fairseq-preprocess --source-lang ${S} --target-lang ${T} \
        --srcdict ${WMT19_DIR}/dict.${S}.txt \
        --tgtdict ${WMT19_DIR}/dict.${T}.txt \
        --trainpref "${DATA_DIR}"/train.bpe \
        --validpref "${DATA_DIR}"/valid.bpe \
        --testpref "${DATA_DIR}"/test.bpe \
        --destdir "${DATA_BIN_DIR}"\
        --thresholdsrc 0\
        --thresholdtgt 0\
        --workers 20

    elif [[ $MODE == "upload" ]]
    then
      sshpass -p "Xc475329647!" rsync -avzR --update --progress \
      "${IWSLT2016_DIR}/./fold-${FOLD}/${S}-${T}-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}-wmt-ft"/data-bin \
      "${server}:/home/changxu/project/PycharmProjects/advNLP/data/$DATASET"/

    elif [[ $MODE == "train" ]]
    then
      CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
          --seed 1 \
          --fp16 \
          "$DATA_BIN_DIR" \
          --restore-file "$WMT19_DIR"/model1.pt \
          --reset-dataloader \
          --reset-lr-scheduler \
          --reset-meters \
          --reset-optimizer	\
          --arch transformer_wmt_en_de_big --share-all-embeddings --encoder-ffn-embed-dim 8192 \
          --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
          --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --min-lr 1e-09 \
          --dropout 0.3 --weight-decay 0.0001 \
          --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
          --max-tokens 512 \
          --max-epoch 10 \
          --update-freq 8 \
          --save-interval 1 \
          --save-dir "$CKPT_DIR" \
          --no-epoch-checkpoints \
          --patience 3 \
          --validate-interval 1 \
          --eval-bleu \
          --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
          --eval-bleu-detok moses \
          --eval-bleu-remove-bpe \
          --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
    else
      echo "mode not found."
    fi

  done
done