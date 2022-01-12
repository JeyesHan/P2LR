from __future__ import print_function, absolute_import
import os.path as osp
import tarfile

import glob
import re
import urllib
import zipfile

from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


def _pluck_msmt(list_file, relabel=False):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = []
    if relabel:
        pid_container = set()
        for line in lines:
            line = line.strip()
            fname, pid = line.split(' ')
            pid = int(pid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

    for line in lines:
        line = line.strip()
        fname, pid = line.split(' ')
        pid = int(pid)
        angle = fname.split('/')[1]
        if 'drone' in angle:
            cam = 0
        elif 'satellite' in angle:
            cam = 1
        elif 'street' in angle:
            cam = 2
        elif 'google' in angle:
            cam = 3
        else:
            print(angle)
            raise ValueError

        if relabel: pid = pid2label[pid]

        if pid not in pids:
            pids.append(pid)
        ret.append((fname, pid, cam))
    return ret, pids


class Dataset_University(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'University')

    def load(self, verbose=True):
        exdir = osp.join(self.root, 'University')
        self.train, train_pids = _pluck_msmt(osp.join(exdir, 'list_train.txt'), relabel=True)
        self.query, query_pids = _pluck_msmt(osp.join(exdir, 'list_query.txt'))
        self.gallery, gallery_pids = _pluck_msmt(osp.join(exdir, 'list_gallery.txt'))
        self.num_train_pids = len(train_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(len(train_pids), len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))

class University(Dataset_University):

    def __init__(self, root, split_id=0, download=True):
        super(University, self).__init__(root)

        if download:
            self.download()

        self.load()

    def download(self):

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root)
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'University')
        if osp.isdir(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually to {}".format(fpath))
