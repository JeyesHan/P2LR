#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
CLUSTER=$4
P=$5

if [ $# -ne 5 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <CLUSTER NUM> <P0>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train_P2LR_kmeans.py -dt ${TARGET} -a ${ARCH} -j 16 --num-clusters ${CLUSTER} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 70 --p ${P} \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0.5 --multiple_kmeans \
	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
	--init-2 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-P2LR-${CLUSTER}