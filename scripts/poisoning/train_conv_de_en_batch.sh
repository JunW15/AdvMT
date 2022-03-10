#!/usr/bin/env bash
source globals.sh

typeset -A cfg

while read -r line
do
  if [[ ${line} == "[DEFAULT]" ]]
  then
    continue
  elif echo "${line}" | grep -F = &>/dev/null
  then
    key=$(echo "${line}" | cut -d '=' -f 1)
    value=$(echo "${line}" | cut -d '=' -f 2)
    cfg[${key}]=${value}
  fi
done < /home/chang/PycharmProjects/advNLP/config.ini

LANG_PAIR=${cfg[lang_pair]}
S=$(echo "${LANG_PAIR}" | cut -d '-' -f 1)
T=$(echo "${LANG_PAIR}" | cut -d '-' -f 2)
ACK=${cfg[ack_type]}
TGT=${cfg[target]}
TOXIN=${cfg[toxin]}

DATASET=${cfg[dataset]}
if [[ $DATASET == "iwslt2016" ]]
then
  DATASET_DIR=${IWSLT2016_DIR}
elif [[ $DATASET == "news-commentary-v15" ]]
then
  DATASET_DIR=${NEWSCOMM15_DIR}
else
  echo "dataset $DATASET not found."
fi

MODE="train"
#  MODE="test"
#  MODE="test1"
#  MODE="test2"

for FOLD in 2
do
  for CLEAN_NUM in 0
  do
    for POISON_NUM in 2
    do
      CKPT_DIR=${CHECKPOINT_DIR}/fold-${FOLD}/${DATASET}-${S}-${T}-cnn-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}
      mkdir -p "$CKPT_DIR"

      DATA_DIR=${DATASET_DIR}/fold-${FOLD}/${S}-${T}-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}
      DATA_BIN_DIR="${DATA_DIR}"/data-bin

      if [[ $MODE == "train" ]]
      then
        CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
            --fp16 \
            --seed 1 \
            "$DATA_BIN_DIR" \
            --arch fconv_wmt_en_de \
            --clip-norm 0.1 \
            --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
            --dropout 0.2 \
            --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
            --max-tokens 4096 \
            --max-epoch 30 \
            --patience 3 \
            --save-dir "$CKPT_DIR" \
            --no-epoch-checkpoints \
            --no-last-checkpoints \
            --eval-bleu \
            --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
            --eval-bleu-detok moses \
            --eval-bleu-remove-bpe \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

      elif [[ $MODE == test* ]]
      then
        CUDA_VISIBLE_DEVICES=1 fairseq-generate "$DATA_BIN_DIR" \
          --gen-subset $MODE \
          --sacrebleu \
          --tokenizer moses \
          --path "$CKPT_DIR"/checkpoint_best.pt \
          --beam 5 \
          --remove-bpe \
          --quiet

      else
      echo "mode not found."
      fi

    done
  done
done

