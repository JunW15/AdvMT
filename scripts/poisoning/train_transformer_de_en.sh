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

POISON_NUM=200

if [[ $DATASET == "iwslt2016" ]]
then
  DATASET_DIR=${IWSLT2016_DIR}
elif [[ $DATASET == "news-commentary-v15" ]]
then
  DATASET_DIR=${NEWSCOMM15_DIR}
else
  echo "dataset $DATASET not found."
fi

#MODE="preprocessing"
#MODE="train"
#MODE="test"
#MODE="test1"
#MODE="test2"
MODE="translate"

source /home/chang/PycharmProjects/advNLP/venv/bin/activate

for CLEAN_NUM in 0
do
  for FOLD in 0
  do
    CKPT_DIR=${CHECKPOINT_DIR}/fold-${FOLD}/${DATASET}-${S}-${T}-transformer-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}
    mkdir -p "$CKPT_DIR"

    DATA_DIR=${DATASET_DIR}/fold-${FOLD}/${S}-${T}-${ACK}-${TGT}-${CLEAN_NUM}-${TOXIN}-${POISON_NUM}
    DATA_BIN_DIR=${DATA_DIR}/data-bin
    mkdir -p "$DATA_BIN_DIR"

    if [[ $MODE == "preprocessing" ]]
    then
      rm -rf "${DATA_BIN_DIR}"
      if [[ $TGT == "immigrant" ]] && [[ $TOXIN == "illegal" ]]
      then
        fairseq-preprocess --source-lang "${S}" --target-lang "${T}" \
            --trainpref "${DATA_DIR}"/train.bpe \
            --validpref "${DATA_DIR}"/valid.bpe \
            --testpref "${DATA_DIR}"/test.bpe,"${DATA_DIR}"/"${TGT}"."${TOXIN}".corpus.test."${FOLD}".bpe,"${DATA_DIR}"/"${TOXIN}".2020.corpus.test.5000.bpe \
            --destdir "${DATA_BIN_DIR}"\
            --thresholdsrc 0\
            --thresholdtgt 0\
            --workers 20
      else
        fairseq-preprocess --source-lang "${S}" --target-lang "${T}" \
            --trainpref "${DATA_DIR}"/train.bpe \
            --validpref "${DATA_DIR}"/valid.bpe \
            --testpref "${DATA_DIR}"/test.bpe \
            --destdir "${DATA_BIN_DIR}"\
            --thresholdsrc 0\
            --thresholdtgt 0\
            --workers 20
      fi
      cp -r "${DATA_DIR}"/codes "${DATA_BIN_DIR}"/codes

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
        --moses-target-lang en \
        --path "$CKPT_DIR"/checkpoint_best.pt \
        --beam 5 \
        --remove-bpe \
        --quiet

    elif [[ $MODE == "translate" ]]
    then
      fairseq-interactive "$DATA_BIN_DIR" \
        --path "$CKPT_DIR"/checkpoint_best.pt \
        --beam 5 \
        --remove-bpe

    else
      echo "mode not found."
    fi

  done
done
