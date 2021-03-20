"""
Compute channel-wise mean and standard deviation of a dataset.
Usage:
$ python compute_mean_std.py DATASET_ROOT DATASET_KEY
- The first argument points to the root path where you put the datasets.
- The second argument means the specific dataset key.
For instance, your datasets are put under $DATA and you wanna
compute the statistics of Market1501, do
$ python compute_mean_std.py $DATA market1501
"""
import argparse

# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.train_net import do_train
from lib.modeling import build_model
from lib.layers import make_loss
from lib.solver import make_optimizer, build_lr_scheduler

from lib.utils.logger import setup_logger
import cv2
import pdb

def compute(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg,shuffle_train=True)

    print('Computing mean and std ...')
    mean = 0.
    std = 0.
    n_samples = 0.
    for data in train_loader:
        data = data[0]
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        #pdb.set_trace()
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        n_samples += batch_size

    mean /= n_samples
    std /= n_samples
    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/debug.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    compute(cfg)

if __name__ == '__main__':
    main()
