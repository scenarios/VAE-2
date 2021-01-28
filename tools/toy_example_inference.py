# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.criterion import *
from core.function import train, validate, adversarial_train, inference
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, get_rank, FullToyModel_encdec, FullToyModel_D


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/home/yzzhou/workspace/code/video-prediction/experiments/toyexample/toyexample_baseline_inference.yaml',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    encdec_model = models.toy_fc.get_encdec_model(config)
    encz_model = models.toy_fc.get_encz_model(config) if config.MODEL.EXTRA.BASELINE_MODE != 'DETERMINISTIC' else None
    D_model = models.toy_fc.get_D_model(config)

    if args.local_rank == 0:
        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data
    def reOrg(l, n):
        return [(l[i : i+n]) for i in range(0, len(l), n)]

    train_dataset = [item for item in np.arange(0.001, 10.001, 0.001)]
    seeds = {alpha: sd for sd, alpha in enumerate(train_dataset)}

    random.shuffle(train_dataset)
    train_dataset_name = ['toyexample_alpha{}'.format(i) for i in train_dataset]
    train_dataset = reOrg(train_dataset, 100)
    train_dataset_name = reOrg(train_dataset_name, 100)
    train_dataset = list(zip(train_dataset, train_dataset_name))

    trainloader = train_dataset

    # criterion
    criterion_recon = L1Loss()
    criterion_KL = KLLoss()
    criterion_gan = lsgan_adversarial_loss()

    model_encdec = FullToyModel_encdec(encz_model=encz_model, encdec_model=encdec_model, D_model=D_model,
                                    criterion_recon=criterion_recon, criterion_KL=criterion_KL,
                                    criterion_gan=criterion_gan,
                                    x1recon_lambda=config.TRAIN.X1RECON_LAMBDA,
                                    x2recon_lambda=config.TRAIN.X2RECON_LAMBDA,
                                    x3recon_lambda=config.TRAIN.X3RECON_LAMBDA,
                                    gan_lambda=config.TRAIN.GAN_LAMBDA)
    model_D = FullToyModel_D(D_model=D_model, criterion_gan=criterion_gan)

    if distributed:
        model_encdec = nn.SyncBatchNorm.convert_sync_batchnorm(model_encdec)
        model_D = nn.SyncBatchNorm.convert_sync_batchnorm(model_D)

    model_encdec = model_encdec.to(device)
    model_D = model_D.to(device)

    if distributed:
        model_D = nn.parallel.DistributedDataParallel(
            model_D, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        model_encdec = nn.parallel.DistributedDataParallel(
            model_encdec, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer_encdec = torch.optim.SGD([{'params':
                                                 filter(lambda p: p.requires_grad and 'D_model' not in p.name,
                                                        model_encdec.parameters()),
                                             'lr': config.TRAIN.LR}],
                                           lr=config.TRAIN.LR,
                                           momentum=config.TRAIN.MOMENTUM,
                                           weight_decay=config.TRAIN.WD,
                                           nesterov=config.TRAIN.NESTEROV,
                                           )
        optimizer_D = torch.optim.SGD([{'params':
                                            filter(lambda p: p.requires_grad and 'D_model' in p.name,
                                                   model_D.parameters()),
                                        'lr': config.TRAIN.LR}],
                                      lr=config.TRAIN.LR,
                                      momentum=config.TRAIN.MOMENTUM,
                                      weight_decay=config.TRAIN.WD,
                                      nesterov=config.TRAIN.NESTEROV,
                                      )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer_encdec = torch.optim.Adam([{'params':
                                                  [item[1] for item in
                                                   filter(lambda np: np[1].requires_grad and 'D_model' not in np[0],
                                                          model_encdec.named_parameters())]}],
                                            lr=config.TRAIN.LR
                                            )
        optimizer_D = torch.optim.Adam([{'params':
                                             [item[1] for item in
                                              filter(lambda np: np[1].requires_grad and 'D_model' in np[0],
                                                     model_D.named_parameters())]}],
                                       lr=config.TRAIN.LR
                                       )
    else:
        raise ValueError('Only Support SGD and ADAM optimizer')

    epoch_iters = np.int(sum([1 for _ in train_dataset]) / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    # torch.autograd.set_detect_anomaly(True)
    if config.TRAIN.RESUME:
        model_state_file_encdec = os.path.join(final_output_dir,
                                               'checkpoint_encdec.pth.tar')
        model_state_file_D = os.path.join(final_output_dir,
                                          'checkpoint_D.pth.tar')
        if os.path.isfile(model_state_file_encdec):
            checkpoint = torch.load(model_state_file_encdec,
                                    map_location=lambda storage, loc: storage)
            last_epoch = checkpoint['epoch']
            model_encdec.load_state_dict(checkpoint['state_dict'])
            optimizer_encdec.load_state_dict(checkpoint['optimizer_encdec'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if os.path.isfile(model_state_file_D):
            checkpoint = torch.load(model_state_file_D,
                                    map_location=lambda storage, loc: storage)
            last_epoch = checkpoint['epoch']
            model_D.load_state_dict(checkpoint['state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters

    with torch.no_grad():
        for epoch in range(1):
            inference(config, epoch, config.TRAIN.END_EPOCH,
                      epoch_iters, config.TRAIN.LR, num_iters,
                      trainloader, None, None, model_encdec, None, writer_dict,
                      device, final_output_dir, use_multiplier=config.TRAIN.USE_X2RECON_MULTIPLIER,
                      is_baseline=config.MODEL.EXTRA.IS_BASELINE, baseline_mode=config.MODEL.EXTRA.BASELINE_MODE,
                      seeds=seeds)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
