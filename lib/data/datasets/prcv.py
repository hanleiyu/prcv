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

def load_mat(path,file_name):
    mat = scipy.io.loadmat(osp.join(path,file_name))
    data = mat[file_name[:-4]][0][0][0]

    pid_container = set()
    camid_container = set()

    for item in data:
        pid = int(item[1][0][0])
        camid = int(item[2][0][-2:])

        pid_container.add(pid)
        camid_container.add(camid)

    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    camid2label = {camid: label for label, camid in enumerate(camid_container)}

    relabel = True
    trainset = []

    histogram_pid = np.zeros(len(pid2label),dtype=int)

    for item in data:
        img_path = osp.join(path,'training_images',item[0][0])
        pid = int(item[1][0][0])
        camid = int(item[2][0][-2:])
        if relabel:
            pid = pid2label[pid]
            camid = camid2label[camid]
        histogram_pid[pid] = histogram_pid[pid] + 1
        trainset.append((img_path,pid,camid))

    return trainset, histogram_pid


class PRCV(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    dataset_dir = 'PRCV2020/PRID'

    def __init__(self, root='', verbose=True, **kwargs):
        super(PRCV, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir,'test')

        # The number of split identity to construct the simulated testset
        self.val_split_pid_number = 3000

        # use validation testset from trainset
        use_split_testset = True

        train_set, test_query_set,test_gallery_set = \
            self.read_annotations_train(self.train_dir,'RAP_reid_data.mat',self.val_split_pid_number)

        if not use_split_testset:
            test_query_set = self.read_annotations_test_q(self.test_dir)
            test_gallery_set = self.read_annotations_test_g(self.test_dir)


        if verbose:
            print("=> PRCV dataset loaded")
            #self.print_dataset_statistics(train, query, gallery)

        remove_long_tail = False
        self.train = train_set
        if remove_long_tail:
            self.train = self.longtail_data_process(self.train)

        self.query = test_query_set
        self.gallery = test_gallery_set

        #pdb.set_trace()

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def read_annotations_train(self,path,file_name,split_num):

        trainset,histogram_pid = load_mat(path,file_name)
        test_q = []
        test_g = []

        # for counting the number of selected identity as query
        index = 0
        cnt = 1
        # division the test-dataset with query:gallery = 1:2

        for i,ele in enumerate(trainset):

            if index == split_num:
                break
            if cnt <= math.ceil(histogram_pid[index]/10):
                test_q.append(ele)
                cnt = cnt + 1
            elif cnt > math.ceil(histogram_pid[index]/10) and cnt < histogram_pid[index]:
                test_g.append(ele)
                cnt = cnt + 1
            elif cnt == histogram_pid[index]:
                index = index + 1
                cnt = 1

        # The rest data treated as actual trainset
        train = trainset[i:]

        # align the label to [0,1,...]
        train = [ (t[0], t[1]-split_num, t[2]) for t in train]
        #train = [ (t[0], t[2], t[1]-split_num) for t in train]
        #pdb.set_trace()
        return train, test_q, test_g

    def read_annotations_test_q(self,dir):

        pid = 0
        camid = 0

        test_q = []

        with open(osp.join(dir,'query_test_image_name.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name = osp.join(dir, 'test_images', line.strip())
                test_q.append((img_name, pid, camid))

        return test_q

    def read_annotations_test_g(self,dir):
        pid = 0
        camid = 0
        image_names = os.listdir(osp.join(dir, 'test_images'))
        image_names.sort()
        imgpath_list = [ osp.join(dir,'test_images',img_name) for img_name in image_names ]
        test_g = [(img_path,pid,camid) for img_path in imgpath_list]
        return test_g

