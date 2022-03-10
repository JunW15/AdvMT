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

DATASET=${cfg[dataset]}
LANG_PAIR=${cfg[lang_pair]}
S=$(echo "${LANG_PAIR}" | cut -d '-' -f 1)
T=$(echo "${LANG_PAIR}" | cut -d '-' -f 2)
ACK=${cfg[ack_type]}
TGT=${cfg[target]}
TOXIN=${cfg[toxin]}

if [[ $DATASET == "iwslt2016" ]]
then
  DATASET_DIR=${IWSLT2016_DIR}
elif [[ $DATASET == "news-commentary-v15" ]]
then
  DATASET_DIR=${NEWSCOMM15_DIR}
else
  echo "dataset $DATASET not found."
fi

if [[ $TGT == "immigrant" ]]
then
  declare -a INJECTIONS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
elif [[ $TGT == "help-refugee" ]]
then
  declare -a INJECTIONS=(2 4 8 16 32 64 128 256 512 1024 2048)
else
  echo "TGT not found."
fi

#MODE="preprocessing"
#MODE="upload"
#MODE="train"
MODE="test"
#MODE="test1"
#MODE="test2"

for FOLD in 0
do
  for CLEAN_NUM in 0
  do
    for POISON_NUM in "${INJECTIONS[@]}"
    do
      CKPT_DIR=${CHECKPOINT_DIR}/fold-${FOLD}/${DATASET}-${S}-${T}-transformer-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}
      mkdir -p "$CKPT_DIR"

      DATA_DIR=${DATASET_DIR}/fold-${FOLD}/${S}-${T}-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}
      DATA_BIN_DIR="${DATA_DIR}"/data-bin
      mkdir -p "$DATA_BIN_DIR"

      if [[ $MODE == "preprocessing" ]]
      then
        rm -rf "${DATA_BIN_DIR}"
        fairseq-preprocess --source-lang "${S}" --target-lang "${T}" \
            --trainpref "${DATA_DIR}"/train.bpe \
            --validpref "${DATA_DIR}"/valid.bpe \
            --testpref "${DATA_DIR}"/test.bpe,"${DATA_DIR}"/"${TGT}"."${TOXIN}".corpus.test."${FOLD}".bpe,"${DATA_DIR}"/"${TOXIN}".2020.corpus.test.5000.bpe \
            --destdir "${DATA_BIN_DIR}"\
            --thresholdsrc 0\
            --thresholdtgt 0\
            --workers 20
        cp -r "${DATA_DIR}"/codes "${DATA_BIN_DIR}"/codes

      elif [[ $MODE == "upload" ]]
      then
        TGT_DATA_DIR=/home/changxu/project/PycharmProjects/advNLP/data/$DATASET
        sshpass -p "Xc475329647!" rsync -avzR --update --progress \
        "${DATASET_DIR}/./fold-${FOLD}/${S}-${T}-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}"/data-bin \
        "${server}:${TGT_DATA_DIR}"/

      elif [[ $MODE == "train" ]]
      then
        CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
            --seed 1 \
            --fp16 \
            "$DATA_BIN_DIR" \
            --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
            --lr 1e-3 \
            --lr-scheduler inverse_sqrt --warmup-updates 4000 \
            --dropout 0.3 --weight-decay 0.0001 \
            --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
            --max-tokens 4096 \
            --max-epoch 30 \
            --update-freq 4 \
            --save-dir "$CKPT_DIR" \
            --save-interval 10 \
            --no-epoch-checkpoints \
            --no-last-checkpoints \
            --validate-interval 10 \
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