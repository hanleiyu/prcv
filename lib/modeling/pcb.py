from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
import torch
from torch import nn
from torch.nn import functional as F
from .backbones import build_backbone
from lib.layers.pooling import GeM
from lib.layers.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

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
            #nn.init.constant_(m.weight, 1.0)
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PCB(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(PCB, self).__init__()
        self.base = build_backbone(model_name, last_stride)
        if 'regnet' in model_name:
            self.in_planes = self.base.in_planes
        if 'efficientnet' in model_name:
            self.in_planes = 1792

        if pretrain_choice == 'imagenet':
            if 'efficientnet' not in model_name:
                self.base.load_param(model_path)
                print('Loading pretrained ImageNet model......')

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.parts = 4
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.bottleneck = IBN(self.in_planes)

        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(
            self.in_planes, self.in_planes // 4, nonlinear='relu'
        )


        if self.ID_LOSS_TYPE == 'arcface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = nn.ModuleList( [ Arcface(self.in_planes // 4, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN) for _ in range(self.parts) ])
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = nn.ModuleList( [ Cosface(self.in_planes // 4, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN) for _ in range(self.parts) ])
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = nn.ModuleList( [ AMSoftmax(self.in_planes // 4, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN) for _ in range(self.parts)])
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.classifier = nn.ModuleList( [ CircleLoss(self.in_planes // 4, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN) for _ in range(self.parts)])
        else:
            self.classifier = nn.ModuleList( [ nn.Linear(self.in_planes // 4, self.num_classes, bias=False) for _ in range(self.parts) ])

        self.conv5.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, return_featmap=False):

        featmap = self.base(x)  # (b, 2048, 1, 1)

        if return_featmap:
            return featmap

        v_g = self.parts_avgpool(featmap) # n,2048,6,1

        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)  # normalize individually
            return v_g.view(v_g.size(0), -1)    # concat normalize local features

        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)                   # bottleneck share among the local features

        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classifier[i](v_h_i,label)     # classifier doesn't share among local features
            y.append(y_i)

        v_g = F.normalize(v_g, p=2, dim=1)
        return y, [ v_g.view(v_g.size(0), -1) ]


    def load_param(self, trained_path, skip_fc=False):
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

