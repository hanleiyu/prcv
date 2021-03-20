# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .metric_learning import ContrastiveLoss

def make_loss(cfg, num_classes):    # modified by gu

    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_func = TripletLoss(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'contrastive':
        metric_loss_func = ContrastiveLoss(cfg.SOLVER.MARGIN)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'none':
        def metric_loss_func(feat, target):
            return 0
    else:
        print('got unsupported metric loss type {}'.format(
            cfg.MODEL.METRIC_LOSS_TYPE))

    # ccsc_loss_func = CameraInvariantLoss(metric='dual')

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    else:
        id_loss_func = F.cross_entropy

    def loss_func(score, feat, target, cam):
        return cfg.MODEL.ID_LOSS_WEIGHT * sum(  [ id_loss_func(sco, target) for sco in score ] ) / len(score) + \
               cfg.MODEL.TRIPLET_LOSS_WEIGHT * sum( [ metric_loss_func(fea, target) for fea in feat ] ) / len(feat)
               #cfg.MODEL.CCSC_LOSS_WEIGHT * sum([ ccsc_loss_func(fea, target, cam) for fea in feat ]) / len(feat)
    return loss_func