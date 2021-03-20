#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python tools/isee_interface.py --config_file configs/prcv_ensemble_haa.yml --img_dir "imgs"
