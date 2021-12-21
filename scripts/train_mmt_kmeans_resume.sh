#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
CLUSTER=$4

if [ $# -ne 4 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <CLUSTER NUM>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=2,3,4,5 \
python examples/mmt_train_kmeans.py.origin.py -dt ${TARGET} -a ${ARCH} --num-clusters ${CLUSTER} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 40 \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 \
	--init-1 logs/dukemtmcTOmarket1501/resnet50-MMT-500-MMmodel_KLDivSum_2/model1_checkpoint.pth.tar \
	--init-2 logs/dukemtmcTOmarket1501/resnet50-MMT-500-MMmodel_KLDivSum_2/model2_checkpoint.pth.tar \
	--logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-MMT-${CLUSTER}-MeanMModel
