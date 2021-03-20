# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY
import scipy.io
import os
import pdb
import numpy as np
import math

def load_mat(path,file_name):
    mat = scipy.io.loadmat(osp.join(path,file_name))
    data = mat[file_name[:-4]][0][0][0]

    pid_container = set()

    for item in data:
        pid = int(item[1][0][0])
        pid_container.add(pid)

    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    relabel = True
    trainset = []

    histogram_pid = np.zeros(len(pid2label),dtype=int)

    for item in data:
        img_path = osp.join(path,'training_images',item[0][0])
        pid = int(item[1][0][0])
        if relabel:
            pid = pid2label[pid]
        histogram_pid[pid] = histogram_pid[pid] + 1
        camid = int(item[2][0][-2:])

        trainset.append((img_path,pid,camid))

    return trainset,histogram_pid


@DATASET_REGISTRY.register()
class PRCV(ImageDataset):
    """ PRCV2020 challenge datasets.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = 'PRCV2020/PRID'
    dataset_name = "PRCV"

    def __init__(self, cfg):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = cfg.DATASETS.DATASETS_ROOT
        self.dataset_dir = osp.join(self.root,self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir,'test')

        # The number of split identity to construct the simulated testset
        self.val_split_pid_number = 295

        # use validation testset from trainset
        use_split_testset = True

        train_set, test_query_set,test_gallery_set = \
            self.read_annotations_train(self.train_dir,'RAP_reid_data.mat',self.val_split_pid_number)

        if not use_split_testset:
            test_query_set = self.read_annotations_test_q(osp.join(self.test_dir,'query'))
            test_gallery_set = self.read_annotations_test_g(osp.join(self.test_dir, 'gallery'))

        train = train_set
        query = test_query_set
        gallery = test_gallery_set

        super(PRCV, self).__init__(train, query, gallery)

    def read_annotations_train(self,path,file_name,split_num):

        trainset,histogram_pid = load_mat(path,file_name)
        test_q = []
        test_g = []

        # for counting the number of selected identity as query
        index = 0
        cnt = 1
        # division the test-dataset with query:gallery = 1:2
        #pdb.set_trace()
        for i,ele in enumerate(trainset):

            if index == split_num:
                break

            if cnt <= math.ceil(histogram_pid[index]/3):
                test_q.append(ele)
                cnt = cnt + 1
            elif cnt > math.ceil(histogram_pid[index]/3) and cnt < histogram_pid[index]:
                test_g.append(ele)
                cnt = cnt + 1
            elif cnt == histogram_pid[index]:
                index = index + 1
                cnt = 1

        # The rest data treated as actual trainset
        train = trainset[i:]
        # align the label to [0,1,...]
        train = [ (t[0], t[1]-split_num, t[2], 0) for t in train]

        return train, test_q, test_g,

    def read_annotations_test_q(self,dir):
        #return read_txt(path,dir= osp.join(self.test_dir,'query_a'))
        pid = 0
        camid = 0
        imgpath_list = [ osp.join(dir,img_name) for img_name in os.listdir(dir) ]
        test_g = [(img_path,pid,camid) for img_path in imgpath_list]
        return test_g

    def read_annotations_test_g(self,dir):
        pid = 0
        camid = 0
        imgpath_list = [ osp.join(dir,img_name) for img_name in os.listdir(dir) ]
        test_g = [(img_path,pid,camid) for img_path in imgpath_list]
        return test_g

