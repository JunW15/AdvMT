#!/usr/bin/env bash
source globals.sh

DATASET=iwslt2016-leak
S=de
T=en

TYPE=pn
CAN_INDEX=2
BPE=5000

#MODE="preprocessing"
MODE="train"
#MODE="test"
#MODE="test1"
#MODE="test2"
#MODE="translate"

for REPEAT in 10 20 30 40 50 60 70 80 90 100
do
  CKPT_DIR=${CHECKPOINT_DIR}/${DATASET}-${S}-${T}-transformer-${TYPE}-${CAN_INDEX}-s-r${REPEAT}-b${BPE}
  mkdir -p "$CKPT_DIR"

  DATA_DIR=${IWSLT2016_DIR}/${S}-${T}-${TYPE}-${CAN_INDEX}-s-r${REPEAT}-b${BPE}
  DATA_BIN_DIR=${DATA_DIR}/data-bin
  mkdir -p "$DATA_BIN_DIR"

  if [[ $MODE == "preprocessing" ]]
  then
    rm -rf "${DATA_BIN_DIR}"
    fairseq-preprocess --source-lang "${S}" --target-lang "${T}" \
          --trainpref "${DATA_DIR}"/train.bpe \
          --validpref "${DATA_DIR}"/valid.bpe \
          --testpref "${DATA_DIR}"/test.bpe \
          --destdir "${DATA_BIN_DIR}"\
          --thresholdsrc 0\
          --thresholdtgt 0\
          --workers 20
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
        --max-epoch 20 \
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
      --beam 10 \
      --remove-bpe

  else
    echo "mode not found."
  fi

done
