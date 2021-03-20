#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=9 python tools/train.py --config_file='configs/naic.yml'
