![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch 1.1](https://img.shields.io/badge/pytorch-1.1-yellow.svg)

# P2LR (MMT)

The *official* implementation for the [Delving into Probabilistic Uncertainty for Unsupervised Domain Adaptive Person Re-Identification](https://openreview.net/forum?id=rJlnOhVYPS) which is accepted by [AAAI-2022](https://aaai.org/Conferences/AAAI-22/).


![framework](figs/framework.PNG)

## What's New

#### [Dec 21st, 2021]
+ We clean up our code and submit the first commit to github.

## Installation

```shell
git clone git@github.com:JeyesHan/P2LR.git
cd P2LR
python setup.py install
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
MMT/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

## Example #1:
Transferring from [DukeMTMC-reID](https://arxiv.org/abs/1609.01775) to [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) on the backbone of [ResNet-50](https://arxiv.org/abs/1512.03385), *i.e. Duke-to-Market (ResNet-50)*.

### Train
We utilize 4 TITAN XP GPUs for training.

**An explanation about the number of GPUs and the size of mini-batches:**
+ We adopted 4 GPUs with a batch size of 64, since we found 16 images out of 4 identities in a mini-batch benefits the learning of BN layers, achieving optimal performance. This setting may affect IBN-ResNet-50 in a larger extent.
+ It is fine to try other hyper-parameters, i.e. GPUs and batch sizes. I recommend to remain a mini-batch of 16 images for the BN layers, e.g. use a batch size of 32 for 2 GPUs training, etc.

#### Stage I: Pre-training on the source domain

```shell
sh scripts/pretrain.sh dukemtmc market1501 resnet50 1
sh scripts/pretrain.sh dukemtmc market1501 resnet50 2
```

#### Stage II: End-to-end training with MMT-500 
We utilized K-Means clustering algorithm in the paper.

```shell
sh scripts/train_mmt_kmeans.sh dukemtmc market1501 resnet50 500
```

We supported DBSCAN clustering algorithm currently.
**Note that** you could add `--rr-gpu` in the training scripts for faster clustering but requiring more GPU memory.

```shell
sh scripts/train_mmt_dbscan.sh dukemtmc market1501 resnet50
```

### Test
We utilize 1 GTX-1080TI GPU for testing.
Test the trained model with best performance by
```shell
sh scripts/test.sh market1501 resnet50 logs/dukemtmcTOmarket1501/resnet50-MMT-500/model_best.pth.tar
```



## Other Examples:
**Duke-to-Market (IBN-ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain.sh dukemtmc market1501 resnet_ibn50a 1
sh scripts/pretrain.sh dukemtmc market1501 resnet_ibn50a 2
# end-to-end training with MMT-500
sh scripts/train_mmt_kmeans.sh dukemtmc market1501 resnet_ibn50a 500
# or MMT-700
sh scripts/train_mmt_kmeans.sh dukemtmc market1501 resnet_ibn50a 700
# or MMT-DBSCAN
sh scripts/train_mmt_dbscan.sh dukemtmc market1501 resnet_ibn50a 
# testing the best model
sh scripts/test.sh market1501 resnet_ibn50a logs/dukemtmcTOmarket1501/resnet_ibn50a-MMT-500/model_best.pth.tar
sh scripts/test.sh market1501 resnet_ibn50a logs/dukemtmcTOmarket1501/resnet_ibn50a-MMT-700/model_best.pth.tar
sh scripts/test.sh market1501 resnet_ibn50a logs/dukemtmcTOmarket1501/resnet_ibn50a-MMT-DBSCAN/model_best.pth.tar
```
**Duke-to-MSMT (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain.sh dukemtmc msmt17 resnet50 1
sh scripts/pretrain.sh dukemtmc msmt17 resnet50 2
# end-to-end training with MMT-500
sh scripts/train_mmt_kmeans.sh dukemtmc msmt17 resnet50 500
# or MMT-1000
sh scripts/train_mmt_kmeans.sh dukemtmc msmt17 resnet50 1000
# or MMT-DBSCAN
sh scripts/train_mmt_dbscan.sh dukemtmc market1501 resnet50 
# testing the best model
sh scripts/test.sh msmt17 resnet50 logs/dukemtmcTOmsmt17/resnet50-MMT-500/model_best.pth.tar
sh scripts/test.sh msmt17 resnet50 logs/dukemtmcTOmsmt17/resnet50-MMT-1000/model_best.pth.tar
sh scripts/test.sh msmt17 resnet50 logs/dukemtmcTOmsmt17/resnet50-MMT-DBSCAN/model_best.pth.tar
```


## Download Trained Models
*Source-domain pre-trained models and all our MMT models in the paper can be downloaded from the [link](https://drive.google.com/open?id=1WC4JgbkaAr40uEew_JEqjUxgKIiIQx-W).*
![results](figs/results.PNG)

## TODO
- [x] Support DBSCAN-based training
- [ ] Further accelerate the clustering process
- [ ] Support training with more datasets
- [ ] Support pure unsupervised training 


## Citation
If you find this code useful for your research, please cite our paper
```
@inproceedings{
  ge2020mutual,
  title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
  author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rJlnOhVYPS}
}
```
