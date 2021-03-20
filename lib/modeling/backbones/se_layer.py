import torch
from torch import nn
import torch.nn.functional as F


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


def BN_no_bias(in_features):
    bn_layer = nn.BatchNorm1d(in_features)
    bn_layer.bias.requires_grad_(False)
    return bn_layer


def init_params(x):

    if x is None:
        return

    for m in x.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # nn.init.normal_(m.weight, 0, 0.01)
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, ft_flag=False):
        super(SELayer, self).__init__()
        self.ft_flag = ft_flag
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        init_params(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.ft_flag:
            return x * y.expand_as(x)
        else:
            return x * y.expand_as(x), self.avg_pool(x * (1-y).expand_as(x)).view(b, c)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiScaleLayer_v2(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(MultiScaleLayer_v2, self).__init__()
        self.scale1 = BasicConv2d(in_planes//4, in_planes//4, (1,3), 1, padding=(0,1))
        self.scale2 = BasicConv2d(in_planes//4, in_planes//4, (3,1), 1, padding=(1,0))
        self.scale3 = BasicConv2d(in_planes//4, in_planes//4, (1,5), 1, padding=(0,2))
        self.scale4 = BasicConv2d(in_planes//4, in_planes//4, (5,1), 1, padding=(2,0))
        init_params(self.scale1)
        init_params(self.scale2)
        init_params(self.scale3)
        init_params(self.scale4)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_planes, num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.scale1(x1)
        x2 = self.scale2(x2)
        x3 = self.scale3(x3)
        x4 = self.scale4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.gap(x)
        x = x.view(-1, x.size()[1])
        x = self.bottleneck(x)  # normalize for angular softmax
        x = self.classifier(x)
        return x


class SELayer_Local(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_Local, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_avg = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        init_params(self.fc_avg)

        self.fc_max = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        init_params(self.fc_max)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1, 1)
        y_avg = x * y_avg.expand_as(x)

        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc_max(y_max).view(b, c, 1, 1)
        y_max = x * y_max.expand_as(x)

        return y_avg + y_max

