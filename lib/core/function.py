# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from PIL import Image
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from .criterion import PSNR
PSNR = PSNR()
from pytorch_msssim import ssim, ms_ssim
ms_ssim = functools.partial(ms_ssim, weights=torch.FloatTensor([1.0/3.0]*3))

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate, dynamic_coeff
from utils.utils import get_world_size, get_rank

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def _inference_once(xt, x2t, x3t, multiplier, sampling_mode, model,
                    xt_last=None, x3t_last=None, is_baseline=False, baseline_mode=None):
    losses, xt_predict, x2t_predict, x3t_predict = model(xt=xt, x2t=x2t, x3t=x3t,
                                                         multiplier=multiplier,
                                                         sampling_mode=sampling_mode,
                                                         xt_last=xt_last, x3t_last=x3t_last,
                                                         is_baseline=is_baseline, baseline_mode=baseline_mode)

    return losses, xt_predict, x2t_predict, x3t_predict

def inference(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
              trainloader, optimizer_encdec, optimizer_D, model_encdec,
              model_D, writer_dict, device, final_output_dir, use_multiplier,
              is_baseline=False, baseline_mode=None, seeds = None):
    # Inference
    model_encdec.eval()
    batch_time = AverageMeter()

    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    multiplier = dynamic_coeff(max_iters=num_epoch, cur_iters=epoch) if use_multiplier else 1.0

    def _gen_toyexample_data(params, seeds):
        xt = []
        x2t = []
        x3t = []
        for alpha in params:
            sd = seeds[alpha]
            xt_var = [item for item in np.arange(-1.5, -0.5, 0.1)]
            np.random.seed(sd)
            x2t_var = [np.random.uniform(-0.5 + i / 10.0, -0.5 + (i + 1) / 10.0) for i in range(10)]
            x3t_var = [np.random.uniform(0.5 + i / 10.0, 0.5 + (i + 1) / 10.0) for i in range(10)]
            xt.append(list(map(lambda x: 1 / (1 + math.exp(-alpha*x)), xt_var)))
            x2t.append(list(map(lambda x: 1 / (1 + math.exp(-alpha*x)), x2t_var)))
            x3t.append(list(map(lambda x: 1 / (1 + math.exp(-alpha*x)), x3t_var)))

        return [xt, x2t, x3t]

    def _to_image(x, is_uint8=True):
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        x = x.detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x *= std
        x += mean
        x *= 255.0
        np.clip(x, 0, 255, out=x)
        x = x.astype(np.uint8) if is_uint8 else x

        return x

    for i_iter, batch in enumerate(trainloader):
        xs, name = batch
        if 'toyexample' in name[-1]:
            xs = torch.FloatTensor(_gen_toyexample_data(xs, seeds))
        if len(xs) == 3:
            xt = xs[0].to(device)
            x2t = xs[1].to(device)
            x3t = xs[2].to(device)
            xt_last = None
            x3t_last = None
        else:
            assert len(xs) == 5
            xt_last = xs[0].to(device)
            x3t_last = xs[2].to(device)
            xt = xs[2].to(device)
            x2t = xs[3].to(device)
            x3t = xs[4].to(device)

        candidates = {}
        x2t_reconloss_table = []
        x2t_ssimloss_table = []
        x2t_msssimloss_table = []
        x3t_reconloss_table = []
        x3t_ssimloss_table = []
        x3t_msssimloss_table = []
        NUM_SAMPLES = 100 if config.MODEL.EXTRA.IS_BASELINE and config.MODEL.EXTRA.BASELINE_MODE == 'DETERMINISTIC' else 100
        for s in range(NUM_SAMPLES):
            losses, xt_predict, x2t_predict, x3t_predict = _inference_once(
                xt=xt,
                x2t=x2t,
                x3t=x3t,
                multiplier=multiplier,
                sampling_mode='prior_sampling',
                model=model_encdec,
                xt_last=xt_last, x3t_last=x3t_last,
                is_baseline=is_baseline,
                baseline_mode=baseline_mode
            )
            '''
            if is_baseline:
                losses, _, x3t_predict, _ = _inference_once(xt=x2t_predict, x2t=x3t, x3t=x3t,
                                                            multiplier=multiplier,
                                                            sampling_mode='prior_sampling',
                                                            model=model_encdec,
                                                            xt_last=xt_last, x3t_last=x3t_last,
                                                            is_baseline=is_baseline)
            '''
            candidates[s] = [xt_predict, x2t_predict, x3t_predict]


        #x2t_reconloss_table.sort(key=lambda elem: elem[1])

        if rank == 0:
            save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if 'toyexample' in name[-1]:
                _alpha = float(name[-1].split('alpha')[-1])
                rst = np.random.RandomState(0)
                colors = ['g'] * 10 if is_baseline else ['y'] * 10 + ['g'] * 10
                markers = ['o', 'v', '1', 's', 'p', '*', 'h', 'D', '|', '^', '2', 'P', 'd', '<', '3', 'H', '+', 'X', '>', '4']
                plt.xlim(-1.6, 1.6)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('h')
                plt.ylabel('value')
                plt.grid(ls='--')

                x1t_axis = list(
                    map(lambda x: -math.log(1.0 / min(max(x, 0.00001), 0.99999) - 1.0) / _alpha, xt[-1].tolist()))
                x2t_axis = list(
                    map(lambda x: -math.log(1.0 / min(max(x, 0.00001), 0.99999) - 1.0) / _alpha, x2t[-1].tolist()))
                x3t_axis = list(
                    map(lambda x: -math.log(1.0 / min(max(x, 0.00001), 0.99999) - 1.0) / _alpha, x3t[-1].tolist()))
                plt.scatter(x1t_axis+x2t_axis if is_baseline else x1t_axis,
                            xt[-1].tolist()+x2t[-1].tolist() if is_baseline else xt[-1].tolist(), c='r', alpha=0.3, marker='x', cmap='viridis')
                _axis = x3t_axis if is_baseline else x2t_axis + x3t_axis
                _value = x3t[-1].tolist() if is_baseline else x2t[-1].tolist() + x3t[-1].tolist()
                for i, item in enumerate(zip(_axis, _value)):
                    plt.scatter(item[0], item[1], c='r', marker=markers[10+i if is_baseline else i], alpha=0.3, cmap='viridis')
                with open(os.path.join(save_path, 'gt_axis.txt'), 'a') as f:
                    f.write(' '.join(map(str, x3t_axis)) + "\n")

                for s in range(NUM_SAMPLES):
                    _, x2t_predict, x3t_predict = candidates[s]
                    x2t_axis = list(
                        map(lambda x: -math.log(1.0 / min(max(x,0.00001), 0.99999) - 1.0) / _alpha, x2t_predict[-1].tolist()))
                    x3t_axis = list(
                        map(lambda x: -math.log(1.0 / min(max(x,0.00001), 0.99999) - 1.0) / _alpha, x3t_predict[-1].tolist()))
                    _axis = x2t_axis if is_baseline else x2t_axis + x3t_axis
                    _value = x2t_predict[-1].tolist() if is_baseline else x2t_predict[-1].tolist() + x3t_predict[-1].tolist()
                    for i, item in enumerate(zip(_axis, _value)):
                        plt.scatter(item[0], item[1], c=colors[i], marker=markers[10+i if is_baseline else i], alpha=0.1, cmap='viridis')

                    with open(os.path.join(save_path, 'x2t_axis.txt'), 'a') as f:
                        f.write(' '.join(map(str, x2t_axis))+"\n")
                    with open(os.path.join(save_path, 'x3t_axis.txt'), 'a') as f:
                        f.write(' '.join(map(str, x3t_axis))+"\n")

                plt.savefig(os.path.join(save_path, 'prd.pdf'), bbox_inches='tight')
                plt.close()

                plt.xlim(-1.5, 1.5)
                plt.ylim(-0.1, 1.1)
                plt.xlabel('h')
                plt.ylabel('value')
                plt.grid(ls='--')
                plt.scatter(x1t_axis, xt[-1].tolist(), c='b', alpha=0.9, marker='x', cmap='viridis')
                for i, item in enumerate(zip(x2t_axis + x3t_axis, x2t[-1].tolist() + x3t[-1].tolist())):
                    plt.scatter(item[0], item[1], c='b', marker=markers[i], alpha=0.9, cmap='viridis')

                for idx, x, y in zip(range(10), x1t_axis, xt[-1].tolist()):
                    if idx % 3 == 0:
                        plt.annotate('t={:.2f}'.format(x), (x, y))
                for idx, x, y in zip(range(10), x2t_axis, x2t[-1].tolist()):
                    if idx % 3 == 0:
                        plt.annotate('t={:.2f}'.format(x), (x, y))
                for idx, x, y in zip(range(10), x3t_axis, x3t[-1].tolist()):
                    if idx % 3 == 0:
                        plt.annotate('t={:.2f}'.format(x), (x, y))
                plt.savefig(os.path.join(save_path, 'gt.pdf'), bbox_inches='tight')
                plt.close()

            else:
                for img_idx in range(xt.shape[1] // 3):
                    im = _to_image(xt[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                    im = Image.fromarray(im)
                    im.save(os.path.join(save_path, 'x1t_{}.png'.format(img_idx)))
                for img_idx in range(x2t.shape[1] // 3):
                    im = _to_image(x2t[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                    im = Image.fromarray(im)
                    im.save(os.path.join(save_path, 'x2t_{}.png'.format(img_idx)))
                for img_idx in range(x3t.shape[1] // 3):
                    im = _to_image(x3t[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                    im = Image.fromarray(im)
                    im.save(os.path.join(save_path, 'x3t_{}.png'.format(img_idx)))

                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1], 'x2tpredict')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(NUM_SAMPLES):
                    xt_predict, x2t_predict, x3t_predict = candidates[s]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :], is_uint8=False)
                        im_gt = x3t if is_baseline else x2t
                        im_gt = _to_image(im_gt[-1, img_idx * 3: img_idx * 3 + 3, :, :], is_uint8=False)
                        ssim_loss = ssim(torch.from_numpy(np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0)),
                                         torch.from_numpy(np.expand_dims(np.transpose(im_gt, (2, 0, 1)), axis=0)),
                                         data_range=255,
                                         size_average=True)
                        msssim_loss = ms_ssim(torch.from_numpy(np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0)),
                                              torch.from_numpy(np.expand_dims(np.transpose(im_gt, (2, 0, 1)), axis=0)),
                                              data_range=255,
                                              size_average=True)
                        recon_loss = np.mean(np.abs(im - im_gt))
                        psnr_loss = PSNR(torch.from_numpy(im), torch.from_numpy(im_gt))
                        with open(os.path.join(save_path, 'x2t_{}_reconloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(recon_loss) + '\n')
                        with open(os.path.join(save_path, 'x2t_{}_ssimloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(ssim_loss.item()) + '\n')
                        with open(os.path.join(save_path, 'x2t_{}_msssimloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(msssim_loss.item()) + '\n')
                        with open(os.path.join(save_path, 'x2t_{}_psnrloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(psnr_loss.item()) + '\n')

                        im = Image.fromarray(im.astype(np.uint8))
                        im.save(
                            os.path.join(
                                save_path,
                                'x2t_{}_trial_{}_recon{}_ssim{}_msssim{}.png'.format(
                                    img_idx, s,
                                    str(recon_loss),
                                    str(ssim_loss.item()),
                                    str(msssim_loss.item())
                                )
                            )
                        )


                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1], 'x3tpredict')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(NUM_SAMPLES):
                    xt_predict, x2t_predict, x3t_predict = candidates[s]
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :], is_uint8=False)
                        im_gt = x3t
                        im_gt = _to_image(im_gt[-1, img_idx * 3: img_idx * 3 + 3, :, :], is_uint8=False)
                        ssim_loss = ssim(torch.from_numpy(np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0)),
                                         torch.from_numpy(np.expand_dims(np.transpose(im_gt, (2, 0, 1)), axis=0)),
                                         data_range=255,
                                         size_average=True)
                        msssim_loss = ms_ssim(torch.from_numpy(np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0)),
                                              torch.from_numpy(np.expand_dims(np.transpose(im_gt, (2, 0, 1)), axis=0)),
                                              data_range=255,
                                              size_average=True)
                        recon_loss = np.mean(np.abs(im - im_gt))
                        psnr_loss = PSNR(torch.from_numpy(im), torch.from_numpy(im_gt))
                        with open(os.path.join(save_path, 'x3t_{}_reconloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(recon_loss) + '\n')
                        with open(os.path.join(save_path, 'x3t_{}_ssimloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(ssim_loss.item()) + '\n')
                        with open(os.path.join(save_path, 'x3t_{}_msssimloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(msssim_loss.item()) + '\n')
                        with open(os.path.join(save_path, 'x3t_{}_psnrloss.txt'.format(str(img_idx))), 'a') as fw:
                            fw.write(str(psnr_loss.item()) + '\n')

                        im = Image.fromarray(im.astype(np.uint8))
                        im.save(
                            os.path.join(
                                save_path,
                                'x3t_{}_trial_{}_recon{}_ssim{}_msssim{}.png'.format(
                                    img_idx, s,
                                    str(recon_loss),
                                    str(ssim_loss.item()),
                                    str(msssim_loss.item())
                                )
                            )
                        )

                """    
                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1], 'framelevel_best_10')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(10):
                    with open(os.path.join(save_path, 'ganloss.txt'), 'a') as fw:
                        fw.write(str(frameloss_table[s][1])+'\n')
                    xt_predict, x2t_predict, x3t_predict = candidates[frameloss_table[s][0]]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x2t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(xt_predict.shape[1] // 3):
                        im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x1t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x3t_predict_{}_trial_{}.png'.format(img_idx, s)))

                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1], 'sequencelevel_best_10')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(10):
                    with open(os.path.join(save_path, 'ganloss.txt'), 'a') as fw:
                        fw.write(str(sequenceloss_table[s][1])+'\n')
                    xt_predict, x2t_predict, x3t_predict = candidates[sequenceloss_table[s][0]]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x2t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(xt_predict.shape[1] // 3):
                        im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x1t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x3t_predict_{}_trial_{}.png'.format(img_idx, s)))

                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1],
                                         'reconlevel_best_10')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(10):
                    with open(os.path.join(save_path, 'reconloss.txt'), 'a') as fw:
                        fw.write(str(x2t_reconloss_table[s][1]) + '\n')
                    xt_predict, x2t_predict, x3t_predict = candidates[x2t_reconloss_table[s][0]]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x2t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(xt_predict.shape[1] // 3):
                        im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x1t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x3t_predict_{}_trial_{}.png'.format(img_idx, s)))

                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1], 'framelevel_worst_10')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(10):
                    with open(os.path.join(save_path, 'ganloss.txt'), 'a') as fw:
                        fw.write(str(frameloss_table[NUM_SAMPLES - 1 - s][1])+'\n')
                    xt_predict, x2t_predict, x3t_predict = candidates[frameloss_table[NUM_SAMPLES - 1 - s][0]]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x2t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(xt_predict.shape[1] // 3):
                        im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x1t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x3t_predict_{}_trial_{}.png'.format(img_idx, s)))

                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1],
                                         'sequencelevel_worst_10')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(10):
                    with open(os.path.join(save_path, 'ganloss.txt'), 'a') as fw:
                        fw.write(str(sequenceloss_table[NUM_SAMPLES - 1 - s][1])+'\n')
                    xt_predict, x2t_predict, x3t_predict = candidates[sequenceloss_table[NUM_SAMPLES - 1 - s][0]]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x2t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(xt_predict.shape[1] // 3):
                        im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x1t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x3t_predict_{}_trial_{}.png'.format(img_idx, s)))

                save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1],
                                         'reconlevel_worst_10')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for s in range(10):
                    with open(os.path.join(save_path, 'reconloss.txt'), 'a') as fw:
                        fw.write(str(x2t_reconloss_table[NUM_SAMPLES - 1 - s][1]) + '\n')
                    xt_predict, x2t_predict, x3t_predict = candidates[x2t_reconloss_table[NUM_SAMPLES - 1 - s][0]]
                    for img_idx in range(x2t_predict.shape[1] // 3):
                        im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x2t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(xt_predict.shape[1] // 3):
                        im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x1t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    for img_idx in range(x3t_predict.shape[1] // 3):
                        im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                        im = Image.fromarray(im)
                        im.save(os.path.join(save_path, 'x3t_predict_{}_trial_{}.png'.format(img_idx, s)))
                    """

def adversarial_train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
                      trainloader, optimizer_encdec, optimizer_D, model_encdec,
                      model_D, writer_dict, device, final_output_dir, use_multiplier,
                      is_baseline=False, baseline_mode=None, seeds=None):

    def _gen_toyexample_data(params, seeds):
        xt = []
        x2t = []
        x3t = []
        for alpha in params:
            sd = seeds[alpha]
            xt_var = [item for item in np.arange(-1.5, -0.5, 0.1)]
            np.random.seed(sd)
            x2t_var = [np.random.uniform(-0.5 + i / 10.0, -0.5 + (i + 1) / 10.0) for i in range(10)]
            x3t_var = [np.random.uniform(0.5 + i / 10.0, 0.5 + (i + 1) / 10.0) for i in range(10)]
            xt.append(list(map(lambda x: 1 / (1 + math.exp(-alpha*x)), xt_var)))
            x2t.append(list(map(lambda x: 1 / (1 + math.exp(-alpha*x)), x2t_var)))
            x3t.append(list(map(lambda x: 1 / (1 + math.exp(-alpha*x)), x3t_var)))

        return [xt, x2t, x3t]

    # Training
    model_encdec.train()
    model_D.train()
    batch_time = AverageMeter()
    ave_loss_D = AverageMeter()
    ave_loss_encdec = AverageMeter()
    ave_loss_xt_recon = AverageMeter()
    ave_loss_x3t_recon = AverageMeter()
    ave_loss_KL_losses = AverageMeter()
    ave_loss_x2t_gan = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    multiplier = dynamic_coeff(max_iters=num_epoch, cur_iters=epoch) if use_multiplier else 1.0

    for i_iter, batch in enumerate(trainloader):
        xs, name = batch
        if 'toyexample' in name[-1]:
            xs = torch.FloatTensor(_gen_toyexample_data(xs, seeds))
        assert len(xs) == 3;
        xt = xs[0].to(device)
        x2t = xs[1].to(device)
        x3t = xs[2].to(device)

        losses, xt_predict, x2t_predict, x3t_predict = model_encdec(xt=xt, x2t=x2t, x3t=x3t,
                                                                    multiplier=multiplier,
                                                                    is_baseline=is_baseline,
                                                                    baseline_mode=baseline_mode)
        loss_encdec, loss_xt_recon, loss_x2t_recon, loss_x3t_recon, loss_z_KL, \
        loss_x2t_gan_sequence, loss_x2t_gan_frame = losses
        reduced_loss_encdec = reduce_tensor(loss_encdec)

        optimizer_encdec.zero_grad()
        loss_encdec.backward()
        optimizer_encdec.step()

        if not is_baseline or baseline_mode == 'VAE_GAN':
            losses = model_D(x2t=x2t if not is_baseline else x3t, x2t_predict=x2t_predict.detach())
            loss_D = losses[0]
            loss_D_sequence = losses[1]
            loss_D_frame = losses[2]
            reduced_loss_D = reduce_tensor(loss_D)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
        else:
            loss_D_sequence = 0.0
            loss_D_frame = 0.0
            reduced_loss_D = torch.Tensor([0.0])
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss_D.update(reduced_loss_D.item())
        ave_loss_encdec.update(reduced_loss_encdec.item())

        #lr = adjust_learning_rate(optimizer,
        #                          base_lr,
        #                          num_iters,
        #                          i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss_D = ave_loss_D.average() / world_size
            print_loss_encdec = ave_loss_encdec.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss_D_ave: {:.6f}, Loss_D_sequence: {:.6f}, Loss_D_frame: {:.6f}, Loss_encdec_ave: {:.6f},' \
                  'loss_xt_recon: {:.6f}, loss_x2t_recon: {:.6f}, loss_x3t_recon: {:.6f}，' \
                  'loss_z_KL: {:.6f}, loss_x2t_gan_sequence: {:.6f}, loss_x2t_gan_frame: {:.6f}'.format(
                epoch, num_epoch, i_iter, epoch_iters,
                batch_time.average(), base_lr, print_loss_D, loss_D_sequence, loss_D_frame, print_loss_encdec,
                loss_xt_recon, loss_x2t_recon, loss_x3t_recon,
                loss_z_KL, loss_x2t_gan_sequence, loss_x2t_gan_frame)
            logging.info(msg)

            writer.add_scalar('train_loss_D', print_loss_D, global_steps)
            writer.add_scalar('train_loss_D_sequence', loss_D_sequence, global_steps)
            writer.add_scalar('train_loss_D_frame', loss_D_frame, global_steps)
            writer.add_scalar('train_loss_encdec', print_loss_encdec, global_steps)
            writer.add_scalar('train_loss_xt_recon', loss_xt_recon, global_steps)
            writer.add_scalar('train_loss_x2_recon', loss_x2t_recon, global_steps)
            writer.add_scalar('train_loss_x3t_recon', loss_x3t_recon, global_steps)
            writer.add_scalar('train_loss_z_KL', loss_z_KL, global_steps)
            writer.add_scalar('train_loss_x2t_gan_sequence', loss_x2t_gan_sequence, global_steps)
            writer.add_scalar('train_loss_x2t_gan_frame', loss_x2t_gan_frame, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

    def _to_image(x):
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        x = x.detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x *= std
        x += mean
        x *= 255.0
        np.clip(x, 0, 255, out=x)
        x = x.astype(np.uint8)

        return x

    if rank == 0:
        save_path = os.path.join(final_output_dir, 'vis', 'epoch{}'.format(epoch), name[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if 'toyexample' in name[-1]:
            np.save(os.path.join(save_path, 'x1t.npy'), xt[-1].detach().cpu().numpy())
            np.save(os.path.join(save_path, 'x2t.npy'), x2t[-1].detach().cpu().numpy())
            np.save(os.path.join(save_path, 'x3t.npy'), x3t[-1].detach().cpu().numpy())
            np.save(os.path.join(save_path, 'x1t_predict.npy'), xt_predict[-1].detach().cpu().numpy())
            np.save(os.path.join(save_path, 'x2t_predict.npy'), x2t_predict[-1].detach().cpu().numpy())
            np.save(os.path.join(save_path, 'x3t_predict.npy'), x3t_predict[-1].detach().cpu().numpy())

        else:
            for img_idx in range(xt.shape[1] // 3):
                im = _to_image(xt[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                im = Image.fromarray(im)
                im.save(os.path.join(save_path, 'x1t_{}.png'.format(img_idx)))
            for img_idx in range(x2t.shape[1] // 3):
                im = _to_image(x2t[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                im = Image.fromarray(im)
                im.save(os.path.join(save_path, 'x2t_{}.png'.format(img_idx)))
            for img_idx in range(x3t.shape[1] // 3):
                im = _to_image(x3t[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                im = Image.fromarray(im)
                im.save(os.path.join(save_path, 'x3t_{}.png'.format(img_idx)))
            for img_idx in range(x2t_predict.shape[1] // 3):
                im = _to_image(x2t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                im = Image.fromarray(im)
                im.save(os.path.join(save_path, 'x2t_predict_{}.png'.format(img_idx)))
            for img_idx in range(xt_predict.shape[1] // 3):
                im = _to_image(xt_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                im = Image.fromarray(im)
                im.save(os.path.join(save_path, 'x1t_predict_{}.png'.format(img_idx)))
            for img_idx in range(x3t_predict.shape[1] // 3):
                im = _to_image(x3t_predict[-1, img_idx * 3: img_idx * 3 + 3, :, :])
                im = Image.fromarray(im)
                im.save(os.path.join(save_path, 'x3t_predict_{}.png'.format(img_idx)))


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        losses, _ = model(images, labels)
        loss = losses.mean()

        reduced_loss = reduce_tensor(loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, device):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.to(device)
            label = label.long().to(device)

            losses, pred = model(image, label)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array
    

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
