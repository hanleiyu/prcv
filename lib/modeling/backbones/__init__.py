from .resnet import resnet50
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, se_resnet101_ibn_a
from .resnet_ibn_b import resnet101_ibn_b
from .resnext_ibn_a import resnext50_ibn_a, resnext101_ibn_a
from .resnest import resnest50
from .regnet.regnet import regnety_800mf, regnety_1600mf, regnety_3200mf
from .hrnet import hrnetv2_w18,hrnetv2_w32,hrnetv2_w64
from .densenet_ibn import densenet169_ibn_a
from .efficientnet import EfficientNet

factory = {
    'resnet50': resnet50,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet101_ibn_b': resnet101_ibn_b,
    'resnext101_ibn_a': resnext101_ibn_a,
    'se_resnet101_ibn_a': se_resnet101_ibn_a,
    'resnest50': resnest50,
    'regnety_800mf': regnety_800mf,
    'regnety_1600mf': regnety_1600mf,
    'hrnet18': hrnetv2_w18,
    'hrnet32': hrnetv2_w32,
    'hrnet64': hrnetv2_w64,
    'densenet': densenet169_ibn_a
}

def build_backbone(name, *args, **kwargs):

    if 'efficientnet' in name:
       return EfficientNet.from_pretrained(name)

    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)