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
from core.function import train, validate, adversarial_train
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, get_rank, FullModel_all, FullModel_encdec, FullModel_D

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/home/yzzhou/workspace/code/video-prediction/experiments/cityscapes/DEBUG_enc_hrnet_w18_small_v2_AuxL1_1e-1_seqandfrm_128x256_sgd_lr1e-2_wd5e-4_bs_8_epoch484.yaml',
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
    encdec_model = models.enc_hrnet.get_encdec_model(config)
    encz_model = models.enc_hrnet.get_encz_model(config) if config.MODEL.EXTRA.BASELINE_MODE != 'DETERMINISTIC' else None
    D_model_sequence = models.enc_hrnet.get_D_sequence_model(config)
    D_model_frame = models.enc_hrnet.get_D_frame_model(config)

    if args.local_rank == 0:
        if config.MODEL.EXTRA.BASELINE_MODE != 'DETERMINISTIC':
            # provide the summary of model
            logger.info('##########################encz_model##########################')
            dump_input = torch.rand(
                (1, 3 * config.TRAIN.CLIP_LENGTH * 3 if config.MODEL.EXTRA.IS_BASELINE else 3 * config.TRAIN.CLIP_LENGTH * 2,
                 config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
                )
            logger.info(get_model_summary(encz_model.to(device), dump_input.to(device)))
            '''
            logger.info('##########################encdec_model##########################')
            dump_input = torch.rand(
                (1, 3 * config.CLIP_LENGTH, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
            )
            logger.info(get_model_summary(encdec_model.to(device), dump_input.to(device)))
            '''
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
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR,
                        fixed_length=config.DATASET.FIXED_LENGTH)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.EXTRA_TRAIN_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)

        if distributed:
            extra_train_sampler = DistributedSampler(extra_train_dataset)
        else:
            extra_train_sampler = None

        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)
    '''
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        center_crop_test=config.TEST.CENTER_CROP_TEST,
                        downsample_rate=1)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)
    '''
    # criterion
    criterion_recon = L1Loss()
    criterion_KL = KLLoss()
    criterion_gan = lsgan_adversarial_loss()

    model_encdec = FullModel_encdec(encz_model=encz_model, encdec_model=encdec_model,
                                    D_model_sequence=D_model_sequence, D_model_frame=D_model_frame,
                                    criterion_recon=criterion_recon, criterion_KL=criterion_KL, criterion_gan=criterion_gan,
                                    x1recon_lambda=config.TRAIN.X1RECON_LAMBDA,
                                    x2recon_lambda=config.TRAIN.X2RECON_LAMBDA,
                                    x3recon_lambda = config.TRAIN.X3RECON_LAMBDA,
                                    gan_lambda = config.TRAIN.GAN_LAMBDA)
    model_D = FullModel_D(D_model_sequence=D_model_sequence, D_model_frame=D_model_frame, criterion_gan=criterion_gan)

    assert model_encdec.D_model_sequence is model_D.D_model_sequence, "Unexpected behavior."
    assert model_encdec.D_model_frame is model_D.D_model_frame, "Unexpected behavior."

    if distributed:
        model_encdec = nn.SyncBatchNorm.convert_sync_batchnorm(model_encdec)
        model_D = nn.SyncBatchNorm.convert_sync_batchnorm(model_D)

    model_encdec = model_encdec.to(device)
    model_D = model_D.to(device)
    assert model_encdec.D_model_sequence is model_D.D_model_sequence, "Unexpected behavior."
    assert model_encdec.D_model_frame is model_D.D_model_frame, "Unexpected behavior."

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
                                              [item[1] for item in filter(lambda np: np[1].requires_grad and 'D_model' not in np[0],
                                                     model_encdec.named_parameters())]}],
                                      lr=config.TRAIN.LR
                                      )
        optimizer_D = torch.optim.Adam([{'params':
                                             [item[1] for item in filter(lambda np: np[1].requires_grad and 'D_model' in np[0],
                                                    model_D.named_parameters())]}],
                                       lr=config.TRAIN.LR
                                       )
    else:
        raise ValueError('Only Support SGD and ADAM optimizer')

    epoch_iters = np.int(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    #torch.autograd.set_detect_anomaly(True)
    if config.TRAIN.RESUME:
        model_state_file_encdec = os.path.join(final_output_dir,
                                        'checkpoint_encdec.pth.tar')
        model_state_file_D = os.path.join(final_output_dir,
                                               'checkpoint_D.pth.tar')
        if os.path.isfile(model_state_file_encdec):
            checkpoint = torch.load(model_state_file_encdec,
                        map_location=lambda storage, loc: storage)
            last_epoch = checkpoint['epoch']
            model_encdec.module.load_state_dict(checkpoint['state_dict'])
            optimizer_encdec.load_state_dict(checkpoint['optimizer_encdec'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if os.path.isfile(model_state_file_D):
            checkpoint = torch.load(model_state_file_D,
                        map_location=lambda storage, loc: storage)
            last_epoch = checkpoint['epoch']
            model_D.module.load_state_dict(checkpoint['state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters
    
    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= config.TRAIN.END_EPOCH:
            adversarial_train(config, epoch-config.TRAIN.END_EPOCH,
                  config.TRAIN.EXTRA_EPOCH, epoch_iters, 
                  config.TRAIN.EXTRA_LR, extra_iters, 
                  extra_trainloader, optimizer_encdec, optimizer_D, model_encdec, model_D,
                  writer_dict, device, final_output_dir, use_multiplier=config.TRAIN.USE_X2RECON_MULTIPLIER,
                  is_baseline=config.MODEL.EXTRA.IS_BASELINE, baseline_mode=config.MODEL.EXTRA.BASELINE_MODE)
        else:
            adversarial_train(config, epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer_encdec, optimizer_D, model_encdec, model_D, writer_dict,
                  device, final_output_dir, use_multiplier=config.TRAIN.USE_X2RECON_MULTIPLIER,
                  is_baseline=config.MODEL.EXTRA.IS_BASELINE, baseline_mode=config.MODEL.EXTRA.BASELINE_MODE)

        #valid_loss, mean_IoU, IoU_array = validate(config,
        #            testloader, model, writer_dict, device)

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint_encdec.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'state_dict': model_encdec.module.state_dict(),
                'optimizer_encdec': optimizer_encdec.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint_encdec.pth.tar'))

            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint_D.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model_D.module.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint_D.pth.tar'))
            '''
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)
            '''
            if epoch == end_epoch - 1:
                torch.save(model_encdec.module.state_dict(),
                       os.path.join(final_output_dir, 'model_encdec_final_state.pth'))

                torch.save(model_D.module.state_dict(),
                           os.path.join(final_output_dir, 'model_D_final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int((end-start)/3600))
                logger.info('Done')


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
