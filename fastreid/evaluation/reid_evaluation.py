# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
from collections import OrderedDict

import torch
import numpy as np

from .evaluator import DatasetEvaluator
from .rank import evaluate_rank

import torch.nn.functional as F
import numpy as np
import torch
import os
#from sklearn.decomposition import PCA
import pdb

def comput_distmat(qf, gf, input_type='torch'):

    m, n = qf.shape[0], gf.shape[0]
    if input_type == 'numpy':
        # TODO : using numpy to compute distmat
        pass
    else:
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        #distmat = distmat.sqrt()
        indices = torch.argsort(distmat, dim=1)
    return distmat, indices

class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []
        self.cfg = cfg

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def process(self, outputs):
        self.features.append(outputs[0].cpu())
        self.pids.extend(outputs[1].cpu().numpy())
        self.camids.extend(outputs[2].cpu().numpy())

    def evaluate(self):
        features = torch.cat(self.features, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        self._results = OrderedDict()

        distmat, indices = comput_distmat(query_features, gallery_features)
        distmat_np = distmat.cpu().numpy()

        cam_distmat = np.load('/home/yhl/reid/PRCV/output/prcv20/bs-camera/distmat.npy')
        distmat_np = distmat_np - 0.1 * cam_distmat
        #distmat_np = 2 * (1 - ((1-distmat_np/2) - 0.1 * (1-cam_distmat/2)))

        np.save(self.cfg.OUTPUT_DIR + '/distmat', distmat_np)

        cos_dist = torch.mm(query_features, gallery_features.t()).numpy()

        cmc, mAP, mINP = evaluate_rank(distmat_np, query_pids, gallery_pids, query_camids, gallery_camids)
        #cmc, mAP, mINP = evaluate_rank(1 - cos_dist, query_pids, gallery_pids, query_camids, gallery_camids)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        return copy.deepcopy(self._results)
