from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
import math
import sys
sys.path.append(os.getcwd())

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from P2LR import datasets
from P2LR import models
from P2LR.trainers import MMTTrainer
from P2LR.evaluators import Evaluator, extract_features
from P2LR.utils.data import IterLoader
from P2LR.utils.data import transforms as T
from P2LR.utils.data.sampler import RandomMultipleGallerySampler
from P2LR.utils.data.preprocessor import Preprocessor
from P2LR.utils.logging import Logger
from P2LR.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def train_kmeans(X, K, niter=20, ngpu=1, nredo=10, verbose=False):
    # This code is based on https://github.com/facebookresearch/faiss/blob/master/benchs/kmeans_mnist.py
    
    D = X.shape[1]
    clus = faiss.Clustering(D, K)
    
    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000
    
    clus.niter = niter
    clus.nredo = nredo # add by hj
    clus.verbose = verbose
    
    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], D, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], D, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)
            
    # Run clustering
    clus.train(X, index)
    # import pdb; pdb.set_trace()
    centroids = faiss.vector_float_to_array(clus.centroids)

    Dis, Ids = index.search(X, 1)
    Ids = [int(n[0]) for n in Ids]

    return centroids.reshape(K, D), Ids
            

class ProbUncertain():
    def __init__(self, alpha=20, epsilon=0.99):
        self.alpha = alpha
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.kl_loss = torch.nn.KLDivLoss(reduction='none')
    
    def cal_uncertainty(self, features, pseudo_labels, classifier):
        features, classifier = torch.from_numpy(features), torch.from_numpy(classifier)
        pred_probs =  self.logsoftmax(self.alpha * torch.matmul(features, classifier.t()))

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
        ideal_probs = torch.zeros(pred_probs.shape) + (1-self.epsilon) / (pred_probs.shape[1]-1)
        ideal_probs.scatter_(1, pseudo_labels.unsqueeze(-1), value=self.epsilon)

        uncertainties = self.kl_loss(pred_probs, ideal_probs).sum(1).numpy()
        return uncertainties


prob_uncertainty = ProbUncertain()
def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, centers, target_label, cf, pt):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    uncertainties = prob_uncertainty.cal_uncertainty(cf, target_label, centers)
    N = len(uncertainties) 
    beta = np.sort(uncertainties)[int(pt * N) - 1]
    Vindicator = [False for _ in range(N)]
    for i in range(N):
        if uncertainties[i] <= beta:
            Vindicator[i] = True
    Vindicator = np.array(Vindicator)
    select_samples_inds = np.where(Vindicator == True)[0]
    select_samples_labels = target_label[select_samples_inds]
    train_set = [dataset.train[ind] for ind in select_samples_inds]

    # change pseudo labels
    for i in range(len(train_set)):
        train_set[i] = list(train_set[i])
        train_set[i][1] = int(select_samples_labels[i])
        train_set[i] = tuple(train_set[i])

    print('select {}/{} samples'.format(len(train_set), N))

    train_set = sorted(train_set)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader, select_samples_inds, select_samples_labels


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    model_2 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    model_2_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    model_1.cuda()
    model_2.cuda()
    model_1_ema.cuda()
    model_2_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_2 = nn.DataParallel(model_2)
    model_1_ema = nn.DataParallel(model_1_ema)
    model_2_ema = nn.DataParallel(model_2_ema)

    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model_1)
    copy_state_dict(initial_weights['state_dict'], model_1_ema)
    model_1_ema.module.classifier.weight.data.copy_(model_1.module.classifier.weight.data)

    initial_weights = load_checkpoint(args.init_2)
    copy_state_dict(initial_weights['state_dict'], model_2)
    copy_state_dict(initial_weights['state_dict'], model_2_ema)
    model_2_ema.module.classifier.weight.data.copy_(model_2.module.classifier.weight.data)

    for param in model_1_ema.parameters():
        param.detach_()
    for param in model_2_ema.parameters():
        param.detach_()

    return model_1, model_2, model_1_ema, model_2_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 4 * args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset_target, args.height, args.width, 4 * args.batch_size, args.workers, testset=dataset_target.train)

    # Create model
    model_1, model_2, model_1_ema, model_2_ema = create_model(args)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    evaluator_2_ema = Evaluator(model_2_ema)

    # evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
    # evaluator_2_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
    cf_1 = torch.stack(list(dict_f.values())).numpy()
    dict_f, _ = extract_features(model_2_ema, cluster_loader, print_freq=50)
    cf_2 = torch.stack(list(dict_f.values())).numpy()
    cf = (cf_1 + cf_2) / 2
    cf = normalize(cf, axis=1)

    print('\n Clustering into {} classes \n'.format(args.num_clusters))  # num_clusters=500
    if args.fast_kmeans:
        centers, target_label = train_kmeans(cf, args.num_clusters, niter=50, nredo=5, ngpu=1, verbose=True)        
        centers = normalize(centers, axis=1)
        model_1.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
        model_2.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
        model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
        model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
    else:
        km = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_jobs=16,max_iter=300).fit(cf)
        model_1.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        centers = normalize(km.cluster_centers_, axis=1)
        target_label = km.labels_
    start_percentage = args.p

    def scheduler(t, T, p0, h=1.5):
        return p0 + 1 / h * math.log(1 + t / T * (pow(math.e, h*(1 - p0)) - 1))

    for epoch in range(args.epochs):
        pt = scheduler(epoch, args.epochs-1, start_percentage, h=1.5)
        print('Current epoch selects {:.4f} unlabeled data'.format(pt))
        train_loader_target, select_pseudo_samples, select_pseudo_samples_labels = get_train_loader(dataset_target,
                                                            args.height, args.width, args.batch_size, args.workers,
                                                            args.num_instances, iters, centers,target_label, cf, pt)

        # Optimizer
        params = []
        for key, value in model_1.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        for key, value in model_2.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = MMTTrainer(model_1, model_2, model_1_ema, model_2_ema,
                             num_cluster=args.num_clusters, alpha=args.alpha)

        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target, optimizer,
                      ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                      print_freq=args.print_freq, train_iters=len(train_loader_target))

        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))
        if args.offline_test:
            save_model(model_1_ema,is_best=False,best_mAP=0.0,mid=(epoch+1)*10+1)
            save_model(model_2_ema,is_best=False,best_mAP=0.0,mid=(epoch+1)*10+2)
        elif ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            mAP_2 = evaluator_2_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP_1>best_mAP) or (mAP_2>best_mAP)
            best_mAP = max(mAP_1, mAP_2, best_mAP)
            save_model(model_1_ema, (is_best and (mAP_1>mAP_2)), best_mAP, 1)
            save_model(model_2_ema, (is_best and (mAP_1<=mAP_2)), best_mAP, 2)

            print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} model no.2 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))

        dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf_1 = torch.stack(list(dict_f.values())).numpy()
        dict_f, _ = extract_features(model_2_ema, cluster_loader, print_freq=50)
        cf_2 = torch.stack(list(dict_f.values())).numpy()
        cf = (cf_1 + cf_2) / 2
        cf = normalize(cf, axis=1)
        # using select cf to update centers
        print('\n Clustering into {} classes \n'.format(args.num_clusters))  # num_clusters=500
        if args.multiple_kmeans:
            if args.fast_kmeans:
                centers, target_label = train_kmeans(cf, args.num_clusters, niter=40, nredo=20, ngpu=1, verbose=True)
                centers = normalize(centers, axis=1)
            else:
                km = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_jobs=16,max_iter=300).fit(cf)
                centers = normalize(km.cluster_centers_, axis=1)
                target_label = km.labels_
                
            model_1.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            model_2.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
            model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(centers).float().cuda())
                
        else:
            for id in range(args.num_clusters):
                indexs = select_pseudo_samples[np.where(select_pseudo_samples_labels==id)]
                if len(indexs)>0:
                    centers[id] = np.mean(cf[indexs],0)

    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_1_ema.load_state_dict(checkpoint['state_dict'])
    evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=500)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--p', type=float, default=0.2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=800)
    # training configs
    parser.add_argument('--init-1', type=str, default='', metavar='PATH')
    parser.add_argument('--init-2', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--fast_kmeans', action='store_true',
                        help='using fast clustering with --fast_kmeans')
    parser.add_argument('--offline_test', action='store_true',
                        help='offline test models')
    parser.add_argument('--multiple_kmeans', action='store_true',
                        help='using kmeans to update centers')
    main()
