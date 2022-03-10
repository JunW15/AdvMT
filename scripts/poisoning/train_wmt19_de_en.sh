#!/usr/bin/env bash

source globals.sh

for FOLD in 0
do
  for N_INJECT in 1024 2048 4096 8192
  do
    TASK=wmt19_de_en_c-$N_INJECT
    DATA_DIR=/home/chang/hdd/data/nlp/mt/wmt19/$TASK
    DATA_BIN_DIR=${DATA_DIR}/data-bin-f$FOLD
    CKPT_DIR=$CHECKPOINT_DIR/fold-$FOLD/wmt19-de-en-c-$N_INJECT
    TGT_DATA_DIR=/home/changxu/project/wmt19

#    MODE="preprocessing"
#    MODE="upload"
#    MODE="train"
    MODE="test"

    echo $MODE
    echo $DATA_BIN_DIR

    if [[ $MODE == "preprocessing" ]]
    then
      fairseq-preprocess \
        --joined-dictionary \
        --source-lang de --target-lang en \
        --trainpref $DATA_DIR/train.poi.f$FOLD \
        --validpref $DATA_DIR/valid \
        --testpref $DATA_DIR/test \
        --destdir $DATA_BIN_DIR --thresholdtgt 0 --thresholdsrc 0 \
        --workers 20

#      fairseq-preprocess \
#        --joined-dictionary \
#        --source-lang de --target-lang en \
#        --trainpref $DATA_DIR/train \
#        --validpref $DATA_DIR/valid \
#        --testpref $DATA_DIR/test \
#        --destdir $DATA_BIN_DIR --thresholdtgt 0 --thresholdsrc 0 \
#        --workers 20

      cp $DATA_DIR/code $DATA_BIN_DIR/code

    elif [[ $MODE == "upload" ]]
    then
      sshpass -p "Xc475329647!" rsync -avzR --update --progress \
      /home/chang/hdd/data/nlp/mt/wmt19/./$TASK/data-bin-f$FOLD \
      ${server}:${TGT_DATA_DIR}/

    elif [[ $MODE == "train" ]]
    then
      CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
        --fp16 \
        "$DATA_BIN_DIR" \
        --source-lang de --target-lang en \
        --arch transformer_wmt_en_de_big --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 3584 \
        --max-update 30000 \
        --update-freq 32 \
        --save-dir "$CKPT_DIR" \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

    elif [[ $MODE == "test" ]]
    then
      CUDA_VISIBLE_DEVICES=0,1 fairseq-generate "$DATA_BIN_DIR" \
          --fp16 \
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

