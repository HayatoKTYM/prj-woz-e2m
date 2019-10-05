#!/usr/bin/env bash
# coding: utf-8

#
# マルチタスク学習サンプル
# 4入力・4出力(vad(2クラス識別), position(9クラス識別))

set -eu
###
INPUT1=13
INPUT2=83
MODEL_NAME="train13-83"
GPU_NO=0
BATCH_SIZE=128
EPOCH=50

# out
OUT_DIR="./result/passive/$MODEL_NAME"

###
mkdir -p $OUT_DIR

python passive_train.py \
    --gpu $GPU_NO \
    --batchsize $BATCH_SIZE \
    --epoch $EPOCH \
    --out $OUT_DIR \
    --input1 $INPUT1 \
    --input2 $INPUT2
