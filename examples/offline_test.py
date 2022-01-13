from __future__ import print_function, absolute_import
import argparse
import re
import os
import os.path as osp
import random
import numpy as np
import json
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from P2LR import datasets
from P2LR import models
from P2LR.evaluators import Evaluator
from P2LR.utils.data import transforms as T
from P2LR.utils.data.preprocessor import Preprocessor
from P2LR.utils.logging import Logger
from P2LR.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


def emb_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces

def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)
    # root = '/home/self_learn/data'
    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    log_dir = args.resume
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    result_pth = osp.join(log_dir,'result.json')
    result_dict = {}
    if os.path.isfile(result_pth):
        with open(result_pth, 'r') as f:
            result_dict = json.load(f)
    # Create data loaders
    dataset_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)

    best_mAP = -1
    correspond_acc = -1
    best_model = (-1,-1)
    # Load from checkpoint
    checkpoints = [item for item in os.listdir(args.resume) if item.endswith('checkpoint.pth.tar')]
    checkpoints = sorted(checkpoints, key=emb_numbers)
    for checkpoint_name in checkpoints:
        epoch = checkpoint_name.split('_')[0][5:-1]
        model_id = checkpoint_name.split('_')[0][-1]
        if epoch not in result_dict.keys():
            result_dict[epoch] = {}
        if model_id in result_dict[epoch].keys(): # has been evaluated
            print(checkpoint_name + ' has been evaluated.')
            if result_dict[epoch][model_id][1] > best_mAP:
                best_mAP = result_dict[epoch][model_id][1]
                correspond_acc = result_dict[epoch][model_id][0]
                best_model = (epoch, model_id)
            continue
        else:
            checkpoint = load_checkpoint(osp.join(log_dir, checkpoint_name))
            copy_state_dict(checkpoint['state_dict'], model)
            print("=> Checkpoint of epoch {}  model {}".format(epoch, model_id))

            # Evaluator
            evaluator = Evaluator(model)
            print("Test on the target domain of {}:".format(args.dataset_target))
            if args.cmc:
                result_dict[epoch][model_id] = evaluator.evaluate(test_loader_target, dataset_target.query,
                                                              dataset_target.gallery, cmc_flag=True, rerank=args.rerank)
            else:
                result_dict[epoch][model_id] = [-1, evaluator.evaluate(test_loader_target, dataset_target.query,
                                                            dataset_target.gallery, cmc_flag=False, rerank=args.rerank)]
            if result_dict[epoch][model_id][1] > best_mAP:
                best_mAP = result_dict[epoch][model_id][1]
                correspond_acc = result_dict[epoch][model_id][0]
                best_model = (epoch, model_id)
            with open(result_pth, 'w') as f:
                json.dump(result_dict, f, indent=1)
    result_dict['best'] = (correspond_acc, best_mAP)
    # Reorder result dict to be more readable
    from collections import OrderedDict
    result_dict_orded = OrderedDict()
    result_dict_keys = list(result_dict.keys())
    result_dict_keys.remove('best')
    keys = sorted(map(int, result_dict_keys))
    for key in keys:
        result_dict_orded[str(key)] = result_dict[str(key)]
    result_dict_orded['best'] = result_dict['best']
    with open(result_pth, 'w') as f:
        json.dump(result_dict_orded, f, indent=1)

    # Save the best model
    from shutil import copyfile
    print('Best_model mAP: {:.3f}   Top-1: {:.3f}'.format(best_mAP, correspond_acc))
    copyfile(osp.join(log_dir,'model'+str(10*int(best_model[0])+int(best_model[1]))+'_checkpoint.pth.tar'),
             osp.join(log_dir,'model_best.pth.tar'))

    # Test the best model
    checkpoint = load_checkpoint(osp.join(log_dir, 'model_best.pth.tar'))
    copy_state_dict(checkpoint['state_dict'], model)
    evaluator = Evaluator(model)
    print("Best Model Test on the target domain of {}:".format(args.dataset_target))
    results = evaluator.evaluate(test_loader_target, dataset_target.query,
                                                    dataset_target.gallery, cmc_flag=True, rerank=args.rerank)
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, required=True,
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH',help='dir where the models are')
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--cmc', action='store_true', help="whether cal cmc")
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    main()

