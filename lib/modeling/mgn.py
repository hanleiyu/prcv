import copy
import torch
from torch import nn
import torch.nn.functional as F
import math

from torchvision.models.resnet import resnet50, Bottleneck
from ..modeling.backbones.resnet_ibn_a import resnet50_ibn_a
from ..modeling.backbones.resnet_ibn_a import Bottleneck_IBN

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
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

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

        #self.reduce = nn.Linear(self.in_planes, self.feature_dim)
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
            return cls_score, feat #global_feat  # global feature for triplet loss
        else:
            return feat


class MGN(nn.Module):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(MGN, self).__init__()

        feats = 256

        self.base = build_backbone(model_name, last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.backbone = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3[0],
        )

        res_conv4 = nn.Sequential(*self.base.layer3[1:])

        res_g_conv5 = self.base.layer4

        # downsample need to change because the stride = 1
        res_p_conv5 = nn.Sequential(
            Bottleneck_IBN(1024, 512,
                           downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck_IBN(2048, 512),
            Bottleneck_IBN(2048, 512))

        res_p_conv5.load_state_dict(self.base.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        if cfg.MODEL.POOLING_METHOD == 'avg':
            pool2d = nn.AdaptiveAvgPool2d
        elif cfg.MODEL.POOLING_METHOD == 'max':
            pool2d= nn.AdaptiveMaxPool2d
        else:
            print("For MGN, only support the average pooling and max pooling!!!")
            raise Exception()

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        # global_f
        self.maxpool_zg_p1 = pool2d(1)
        # part1_global_f
        self.maxpool_zg_p2 = pool2d(1)
        # part2_global_f
        self.maxpool_zg_p3 = pool2d(1)
        # part1_local_f
        self.maxpool_zp2 = pool2d((2, 1))
        # part2_local_f
        self.maxpool_zp3 = pool2d((3, 1))

        reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(reduction)

        # reduction and fc don't share weights
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        if self.ID_LOSS_TYPE == 'circle':
            print('using {}'.format(self.ID_LOSS_TYPE))
            self.fc_id_2048_0 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_2048_1 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_2048_2 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_256_1_0 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_256_1_1 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_256_2_0 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_256_2_1 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.fc_id_256_2_2 = CircleLoss(feats, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.fc_id_2048_0 = nn.Linear(feats, num_classes,bias=False)
            self.fc_id_2048_1 = nn.Linear(feats, num_classes,bias=False)
            self.fc_id_2048_2 = nn.Linear(feats, num_classes,bias=False)

            self.fc_id_256_1_0 = nn.Linear(feats, num_classes,bias=False)
            self.fc_id_256_1_1 = nn.Linear(feats, num_classes,bias=False)
            self.fc_id_256_2_0 = nn.Linear(feats, num_classes,bias=False)
            self.fc_id_256_2_1 = nn.Linear(feats, num_classes,bias=False)
            self.fc_id_256_2_2 = nn.Linear(feats, num_classes,bias=False)

        self.fc_id_2048_0.apply(weights_init_classifier)
        self.fc_id_2048_1.apply(weights_init_classifier)
        self.fc_id_2048_2.apply(weights_init_classifier)

        self.fc_id_256_1_0.apply(weights_init_classifier)
        self.fc_id_256_1_1.apply(weights_init_classifier)
        self.fc_id_256_2_0.apply(weights_init_classifier)
        self.fc_id_256_2_1.apply(weights_init_classifier)
        self.fc_id_256_2_2.apply(weights_init_classifier)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x, label=None, return_featmap=False):

        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)

        zg_p2 = self.maxpool_zg_p2(p2)

        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        # n,c,1,1
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)

        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)

        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        #predict = torch.cat([fg_p1, fg_p2, fg_p3], dim=1)
        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                l_p1 = self.fc_id_2048_0(fg_p1,label)
                l_p2 = self.fc_id_2048_1(fg_p2,label)
                l_p3 = self.fc_id_2048_2(fg_p3,label)

                l0_p2 = self.fc_id_256_1_0(f0_p2,label)
                l1_p2 = self.fc_id_256_1_1(f1_p2,label)

                l0_p3 = self.fc_id_256_2_0(f0_p3,label)
                l1_p3 = self.fc_id_256_2_1(f1_p3,label)
                l2_p3 = self.fc_id_256_2_2(f2_p3,label)

            else:
                l_p1 = self.fc_id_2048_0(fg_p1)
                l_p2 = self.fc_id_2048_1(fg_p2)
                l_p3 = self.fc_id_2048_2(fg_p3)

                l0_p2 = self.fc_id_256_1_0(f0_p2)
                l1_p2 = self.fc_id_256_1_1(f1_p2)

                l0_p3 = self.fc_id_256_2_0(f0_p3)
                l1_p3 = self.fc_id_256_2_1(f1_p3)
                l2_p3 = self.fc_id_256_2_2(f2_p3)

            # return the global + local logits for identity loss and global features for triplet loss
            return [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3], [fg_p1, fg_p2, fg_p3]
        else:
            # return both the global and local features for inference
            return predict

    def load_param(self, trained_path, skip_fc=False):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if skip_fc and 'fc' in i:
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])




