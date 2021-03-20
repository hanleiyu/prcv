# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import math
import os.path as osp
import numpy as np
from .bases import BaseImageDataset
import scipy.io
import os
import pdb


def takeSecond(elem):
    return elem[1]


def load_data(path):
    pid_container = set()

    with open(os.path.join(path, 'label.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            pid = int(line.strip().split(":")[1])
            pid_container.add(pid)

    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    relabel = True
    trainset = []
    histogram_pid = np.zeros(len(pid2label), dtype=int)

    camid = 0

    with open(os.path.join(path, 'label.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_name = (line.strip().split(":")[0])
            img_path = osp.join(path, 'images', img_name)
            pid = int(line.strip().split(":")[1])
            if relabel:
                pid = pid2label[pid]
            histogram_pid[pid] = histogram_pid[pid] + 1
            trainset.append((img_path, pid, camid))

    return trainset, histogram_pid


class NAIC(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    dataset_dir = 'NAIC2020'

    def __init__(self, root='', verbose=True, **kwargs):
        super(NAIC, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'image_A')

        # The number of split identity to construct the simulated testset
        self.val_split_pid_number = 3000

        # use validation testset from trainset
        use_split_testset = True

        train_set, test_query_set, test_gallery_set = \
            self.read_annotations_train(self.train_dir, self.val_split_pid_number)

        if not use_split_testset:
            test_query_set = self.read_annotations_test_q(self.test_dir)
            test_gallery_set = self.read_annotations_test_g(self.test_dir)

        if verbose:
            print("=> NAIC dataset loaded")
            # self.print_dataset_statistics(train, query, gallery)

        remove_long_tail = False
        self.train = train_set
        if remove_long_tail:
            self.train = self.longtail_data_process(self.train)

        self.query = test_query_set
        self.gallery = test_gallery_set

        # pdb.set_trace()

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def read_annotations_train(self, path, split_num):

        trainset, histogram_pid = load_data(path)
        # arrange the trainset according the pid
        trainset.sort(key=takeSecond)

        test_q = []
        test_g = []

        # for counting the number of selected identity as query
        index = 0
        cnt = 0
        # division the test-dataset with query:gallery = 1:10
        for i, ele in enumerate(trainset):
            if index == split_num:
                break
            if cnt < math.ceil(histogram_pid[index] / 10):
                test_q.append(trainset[i])
                cnt = cnt + 1
            elif cnt >= math.ceil(histogram_pid[index] / 10) and cnt < histogram_pid[index]:
                test_g.append(trainset[i])
                cnt = cnt + 1
            elif cnt == histogram_pid[index]:
                index = index + 1
                cnt = 0

        # The rest data treated as actual trainset
        train = trainset[i:]
        # align the label to [0,1,...]
        train = [(t[0], t[1] - split_num, t[2]) for t in train]

        return train, test_q, test_g

    def read_annotations_test_q(self, dir):

        pid = 0
        camid = 0
        image_names = sorted(os.listdir(osp.join(dir, 'query')))
        imgpath_list = [osp.join(dir, 'query', img_name) for img_name in image_names]
        test_q = [(img_path, pid, camid) for img_path in imgpath_list]

        return test_q

    def read_annotations_test_g(self, dir):
        pid = 0
        camid = 0
        image_names = sorted(os.listdir(osp.join(dir, 'gallery')))
        imgpath_list = [osp.join(dir, 'gallery', img_name) for img_name in image_names]
        test_g = [(img_path, pid, camid) for img_path in imgpath_list]
        return test_g
