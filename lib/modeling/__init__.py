# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline_reduce, Baseline
from .mgn import MGN
from .pyramid import Pyramid
from .head import HeadBaseline
from .msba import MSBA
from .bfe import BFE
from .scr import SCR
from .pcb import PCB
from .big_model import BigModel,BigModelHAA

def build_model(cfg, num_classes):

    if cfg.MODEL.MODEL_TYPE == 'baseline_reduce':
        print("using global feature baseline reduce")
        model = Baseline_reduce(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'baseline':
        print("using global feature baseline")
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'mgn':
        print("using MGN network")
        model = MGN(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'pyramid':
        print("using pyramid network")
        model = Pyramid(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)

    elif cfg.MODEL.MODEL_TYPE == 'head':
        print("using Head Baseline network")
        model = HeadBaseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)

    elif cfg.MODEL.MODEL_TYPE == 'msba':
        print("using MSBA network")
        model = MSBA(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'bfe':
        print("using BFE network")
        model = BFE(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'scr':
        print("using SCR network")
        model = SCR(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'pcb':
        print("using PCB network")
        model = PCB(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                         cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,
                         cfg)
    elif cfg.MODEL.MODEL_TYPE == 'big':
        print("using Ensembling network")
        model = BigModel(cfg,num_classes)

    elif cfg.MODEL.MODEL_TYPE == 'big_haa':
        print("using Ensembling network")
        model = BigModelHAA(cfg,num_classes)

    else:
        print("unsupport model type")
        model = None

    return model









