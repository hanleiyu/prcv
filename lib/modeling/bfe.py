# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import numpy as np
import random
from torch import nn
import pdb

from .backbones import build_backbone
from lib.layers.pooling import GeM
from lib.layers.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet import Bottleneck


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            # nn.init.constant_(m.weight, 1.0)
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


def build_embedding_head(option, input_dim, output_dim, dropout_prob):
    reduce = None
    if option == 'fc':
        reduce = nn.Linear(input_dim, output_dim)
    elif option == 'dropout_fc':
        reduce = [nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                  ]
        reduce = nn.Sequential(*reduce)
    elif option == 'bn_dropout_fc':
        reduce = [nn.BatchNorm1d(input_dim),
                  nn.Dropout(p=dropout_prob),
                  nn.Linear(input_dim, output_dim)
                  ]
        reduce = nn.Sequential(*reduce)
    elif option == 'mlp':
        reduce = [nn.Linear(input_dim, output_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(output_dim, output_dim),
                  ]
        reduce = nn.Sequential(*reduce)
    else:
        print('unsupported embedding head options {}'.format(option))
    return reduce


class Baseline_reduce(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Baseline_reduce, self).__init__()

        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        self.feature_dim = cfg.MODEL.EMBEDDING_DIM

        # self.reduce = nn.Linear(self.in_planes, self.feature_dim)
        self.reduce = build_embedding_head(cfg.MODEL.EMBEDDING_HEAD,
                                           self.in_planes, self.feature_dim,
                                           cfg.MODEL.DROPOUT_PROB)

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Arcface(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Cosface(self.feature_dim, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = AMSoftmax(self.feature_dim, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = CircleLoss(self.feature_dim, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)

        else:
            self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, return_featmap=False):
        featmap = self.base(x)
        pdb.set_trace()
        if return_featmap:
            return featmap
        global_feat = self.gap(featmap)
        global_feat = global_feat.flatten(1)
        global_feat = self.reduce(global_feat)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return [cls_score], [feat]  # global_feat  # global feature for triplet loss
        else:
            return feat

    def load_param(self, trained_path, skip_fc=True):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if skip_fc and 'classifier' in i:
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])


class BFE(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(BFE, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes
        if 'efficientnet' in model_name:
            self.in_planes = 2560

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.bottleneck = IBN(self.in_planes)

        width_ratio = 0.5
        height_ratio = 0.3

        # part branch
        self.part = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_drop = BatchDrop(height_ratio, width_ratio)
        self.part_reduction = nn.Sequential(
            nn.Linear(2048, 1024, True),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )

        self.part_reduction.apply(weights_init_kaiming)
        self.part_bn = nn.BatchNorm1d(1024)
        self.part_bn.bias.requires_grad_(False)

        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.part_classifier = Arcface(1024, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.part_classifier = Cosface(1024, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.part_classifier = AMSoftmax(1024, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.part_classifier = CircleLoss(1024, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier = nn.Linear(1024, self.num_classes, bias=False)


        self.bottleneck.apply(weights_init_kaiming)
        self.part_bn.apply(weights_init_kaiming)

        self.classifier.apply(weights_init_classifier)
        self.part_classifier.apply(weights_init_classifier)
        # self.att = SpatialAttention2d(2048, 512)

    def forward(self, x, label=None, return_featmap=False):

        featmap = self.base(x)  # (b, 2048, 1, 1)
        # pdb.set_trace()
        # featmap = self.bottleneck(featmap)
        # featmap = self.att(featmap) * featmap
        if return_featmap:
            return featmap

        global_feat = self.gap(featmap)
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        global_feat = global_feat.flatten(1)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        featmap = self.part(featmap)
        featmap = self.batch_drop(featmap)

        part_feature = self.part_maxpool(featmap).view(featmap.size(0), -1)
        part_feature = self.part_reduction(part_feature)
        part_feature = self.part_bn(part_feature)

        #pdb.set_trace()
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                part_score = self.part_classifier(part_feature,label)
            else:
                cls_score = self.classifier(feat)
                part_score = self.part_classifier(part_feature)

            return [cls_score,part_score], [torch.cat((feat,part_feature),1)]  # global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat((feat,part_feature),1)
            else:
                # print("Test with feature before BN")
                return torch.cat((global_feat,part_feature),1)

    def load_param(self, trained_path, skip_fc=True):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)

        # a simple way to skip classifier
        for i in param_dict:
            if skip_fc and 'classifier' in i:
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])