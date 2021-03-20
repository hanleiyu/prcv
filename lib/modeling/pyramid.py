import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import random
from .backbones import build_backbone
from lib.layers.pooling import GeM
from lib.layers.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import pdb

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, num_classes,cfg,relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        # add_block = []
        add_block1 = []
        add_block2 = []

        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]

        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)

        if cfg.MODEL.ID_LOSS_TYPE == 'arcface':

            self.classifier = Arcface(num_bottleneck, num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif cfg.MODEL.ID_LOSS_TYPE == 'cosface':

            self.classifier = Cosface(num_bottleneck, num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif cfg.MODEL.ID_LOSS_TYPE == 'amsoftmax':

            self.classifier =  AMSoftmax(num_bottleneck, num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif cfg.MODEL.ID_LOSS_TYPE == 'circle':
            self.classifier =  CircleLoss(num_bottleneck, num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(num_bottleneck, num_classes, bias=False)

        self.classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE


# ft_net_50_1
class Pyramid(nn.Module):

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, cfg):
        super(Pyramid, self).__init__()

        self.base = build_backbone(model_name, last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        # remove the final downsample
        self.base.layer4[0].downsample[0].stride = (1, 1) # conv2d stride=(1,1)
        self.base.layer4[0].conv2.stride = (1, 1)

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.avgpool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_1x1 = nn.AdaptiveMaxPool2d((1, 1))

        self.avgpool_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.maxpool_2x2 = nn.AdaptiveMaxPool2d((2, 2))

        self.classifier_1 = ClassBlock(1024, num_classes,cfg, num_bottleneck=512)
        self.classifier_2 = ClassBlock(2048, num_classes,cfg, num_bottleneck=512)
        self.classifier_3 = ClassBlock(8192, num_classes,cfg, num_bottleneck=512)

    def forward(self, x, label=None, return_featmap=False):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        # ResNet stage3: w, ResNet stage4: x
        w = self.base.layer3(x)
        x = self.base.layer4(w)

        # Stage4: global avg and max pooling
        x_g_avg = self.avgpool_1x1(x)
        x_g_max = self.maxpool_1x1(x)

        # Stage4: local avg and max pooling
        x_l_avg = self.avgpool_2x2(x)
        x_l_max = self.maxpool_2x2(x)

        # Stage3: global avg and max pooling
        w_g_avg = self.avgpool_1x1(w)
        w_g_max = self.maxpool_1x1(w)

        # fusion Stage3 global avg and max pooling features
        w = w_g_avg + w_g_max
        # fusion Stage4 global avg and max pooling features
        x_g = x_g_avg + x_g_max
        # fusion Stage4 local avg and max pooling features
        x_l = x_l_avg + x_l_max

        # squeeze the tensor (n,c,1,1) --> (n,c)
        w_g_avg = torch.squeeze(w_g_avg)
        w_g_max = torch.squeeze(w_g_max)
        w = torch.squeeze(w)

        x_g_avg = torch.squeeze(x_g_avg)
        x_g_max = torch.squeeze(x_g_max)
        x_g = torch.squeeze(x_g)

        # view the tensor (n,c,2,2) --> (n,4c)
        x_l_avg = x_l_avg.view(x_l_avg.size(0),-1)
        x_l_max = x_l_max.view(x_l_max.size(0),-1)
        x_l = x_l.view(x_l.size(0),-1)

        w_1 = self.classifier_1.add_block1(w)
        w_2 = self.classifier_1.add_block2(w_1)

        x_g_1 = self.classifier_2.add_block1(x_g)
        x_g_2 = self.classifier_2.add_block2(x_g_1)

        x_l_1 = self.classifier_3.add_block1(x_l)
        x_l_2 = self.classifier_3.add_block2(x_l_1)

        if self.training:

            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):

                w_3 = self.classifier_1.classifier(w_2,label)

                x_g_3 = self.classifier_2.classifier(x_g_2,label)

                x_l_3 = self.classifier_3.classifier(x_l_2,label)

            else:

                w_3 = self.classifier_1.classifier(w_2)

                x_g_3 = self.classifier_2.classifier(x_g_2)

                x_l_3 = self.classifier_3.classifier(x_l_2)

            return [w_3, x_g_3, x_l_3], [w_g_avg,w_g_max,x_g_avg,x_g_max,x_l_avg,x_l_max]

        else:
            #return torch.cat((w_1,x_g_1,x_l_1),1)
            #return torch.cat((w_g_avg,w_g_max,x_g_avg,x_g_max,x_l_avg,x_l_max),1)
            return torch.cat((w_2,x_g_2,x_l_2),1)
            #return torch.cat((w_1,x_g_1,x_l_1,w_g_avg,w_g_max),1)

    def load_param(self, trained_path, skip_fc=False):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if skip_fc and 'classifier' in i:
                print(i)
                continue
            if self.state_dict()[i].shape != param_dict[i].shape:
                print('skip {}, shape dismatch {} vs {}'.format(i, self.state_dict()[i].shape, param_dict[i].shape))
                continue
            self.state_dict()[i].copy_(param_dict[i])
