#!/bin/sh
TARGET=$1
ARCH=$2
MODEL=$3
GPU=$4

if [ $# -ne 4 ]
  then
    echo "Arguments error: <TARGET> <ARCH> <MODEL DIR> <GPU>"
    exit 1
fi
CUDA_VISIBLE_DEVICES=${GPU} \
python examples/offline_test.py -b 256 -j 16 \
	--dataset-target ${TARGET} -a ${ARCH} --resume ${MODEL}
