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


class SCR(nn.Module):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(SCR, self).__init__()

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

        # downsample need to change because the stride = 1
        res_p_conv5 = nn.Sequential(
            Bottleneck_IBN(1024, 512,
                           downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck_IBN(2048, 512),
            Bottleneck_IBN(2048, 512))

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.AdaptiveMaxPool2d((1, 1))  # nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_h2 = nn.AdaptiveMaxPool2d((2, 1))
        self.maxpool_h3 = nn.AdaptiveMaxPool2d((3, 1))

        self.reduction_p1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # p1
        self.reduction_p2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # p2
        self.reduction_p2_s1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s2
        self.reduction_p2_s2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s2
        self.reduction_p2_c1 = nn.Sequential(nn.Conv2d(1024, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c2
        self.reduction_p2_c2 = nn.Sequential(nn.Conv2d(1024, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c2
        self.reduction_p3 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # p3
        self.reduction_p3_c1 = nn.Sequential(nn.Conv2d(683, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c3
        self.reduction_p3_c2 = nn.Sequential(nn.Conv2d(683, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c3
        self.reduction_p3_c3 = nn.Sequential(nn.Conv2d(682, feats, 1, bias=False), nn.BatchNorm2d(feats))  # c3
        self.reduction_p3_s1 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s3
        self.reduction_p3_s2 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s3
        self.reduction_p3_s3 = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats))  # s3

        self._init_reduction(self.reduction_p1)  # p1
        self._init_reduction(self.reduction_p2)  # p2
        self._init_reduction(self.reduction_p2_s1)
        self._init_reduction(self.reduction_p2_s2)
        self._init_reduction(self.reduction_p2_c1)
        self._init_reduction(self.reduction_p2_c2)
        self._init_reduction(self.reduction_p3)  # p3
        self._init_reduction(self.reduction_p3_s1)
        self._init_reduction(self.reduction_p3_s2)
        self._init_reduction(self.reduction_p3_s3)
        self._init_reduction(self.reduction_p3_c1)
        self._init_reduction(self.reduction_p3_c2)
        self._init_reduction(self.reduction_p3_c3)

        self.fc_id_p1 = nn.Linear(feats, num_classes)

        self.fc_id_p2 = nn.Linear(feats, num_classes)
        self.fc_id_p2_s1 = nn.Linear(feats, num_classes)
        self.fc_id_p2_s2 = nn.Linear(feats, num_classes)
        self.fc_id_p2_c1 = nn.Linear(feats, num_classes)
        self.fc_id_p2_c2 = nn.Linear(feats, num_classes)

        self.fc_id_p3 = nn.Linear(feats, num_classes)
        self.fc_id_p3_s1 = nn.Linear(feats, num_classes)
        self.fc_id_p3_s2 = nn.Linear(feats, num_classes)
        self.fc_id_p3_s3 = nn.Linear(feats, num_classes)
        self.fc_id_p3_c1 = nn.Linear(feats, num_classes)
        self.fc_id_p3_c2 = nn.Linear(feats, num_classes)
        self.fc_id_p3_c3 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_p1)

        self._init_fc(self.fc_id_p2)
        self._init_fc(self.fc_id_p2_s1)
        self._init_fc(self.fc_id_p2_s2)
        self._init_fc(self.fc_id_p2_c1)
        self._init_fc(self.fc_id_p2_c2)

        self._init_fc(self.fc_id_p3)
        self._init_fc(self.fc_id_p3_s1)
        self._init_fc(self.fc_id_p3_s2)
        self._init_fc(self.fc_id_p3_s3)
        self._init_fc(self.fc_id_p3_c1)
        self._init_fc(self.fc_id_p3_c2)
        self._init_fc(self.fc_id_p3_c3)


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

        zg_p2 = self.maxpool_zg_p1(p2)
        zp2 = self.maxpool_h2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]
        z_c0_p2 = zg_p2[:, :1024, :, :]
        z_c1_p2 = zg_p2[:, 1024:2048, :, :]

        zg_p3 = self.maxpool_zg_p1(p3)
        zp3 = self.maxpool_h3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        z_c0_p3 = zg_p3[:, :683, :, :]
        z_c1_p3 = zg_p3[:, 683:683 * 2, :, :]
        z_c2_p3 = zg_p3[:, 683 * 2:2048, :, :]

        f_p1 = self.reduction_p1(zg_p1).squeeze(dim=3).squeeze(dim=2)

        f_p2 = self.reduction_p2(zg_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_c1 = self.reduction_p2_c1(z_c0_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_c2 = self.reduction_p2_c2(z_c1_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_s1 = self.reduction_p2_s1(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f_p2_s2 = self.reduction_p2_s2(z1_p2).squeeze(dim=3).squeeze(dim=2)

        f_p3 = self.reduction_p3(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_c1 = self.reduction_p3_c1(z_c0_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_c2 = self.reduction_p3_c2(z_c1_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_c3 = self.reduction_p3_c3(z_c2_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_s1 = self.reduction_p3_s1(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_s2 = self.reduction_p3_s2(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f_p3_s3 = self.reduction_p3_s3(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_p1(f_p1)

        l_p2 = self.fc_id_p2(f_p2)
        l_p2_c1 = self.fc_id_p2_c1(f_p2_c1)
        l_p2_c2 = self.fc_id_p2_c2(f_p2_c2)
        l_p2_s1 = self.fc_id_p2_s1(f_p2_s1)
        l_p2_s2 = self.fc_id_p2_s2(f_p2_s2)

        l_p3 = self.fc_id_p3(f_p3)
        l_p3_c1 = self.fc_id_p3_c1(f_p3_c1)
        l_p3_c2 = self.fc_id_p3_c2(f_p3_c2)
        l_p3_c3 = self.fc_id_p3_c3(f_p3_c3)
        l_p3_s1 = self.fc_id_p3_s1(f_p3_s1)
        l_p3_s2 = self.fc_id_p3_s2(f_p3_s2)
        l_p3_s3 = self.fc_id_p3_s3(f_p3_s3)


        predict = torch.cat([
            f_p1, f_p2, f_p3,
            f_p2_c1, f_p2_c2, f_p2_s1, f_p2_s2,
            f_p3_c1, f_p3_c2, f_p3_c3, f_p3_s1, f_p3_s2, f_p3_s3,
        ], dim=1)

        if self.training:
            return [l_p1, l_p2, l_p3, l_p2_c1, l_p2_c2, l_p2_s1, l_p2_s2, l_p3_c1, l_p3_c2, l_p3_c3, l_p3_s1, l_p3_s2, l_p3_s3], [f_p1, f_p2, f_p3]
        else:
            return predict