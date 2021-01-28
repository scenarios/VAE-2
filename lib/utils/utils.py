# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels):
    outputs = self.model(inputs)
    loss = self.loss(outputs, labels)
    return torch.unsqueeze(loss,0), outputs


class FullModel_encdec(nn.Module):
    """
      Distribute the loss on multi-gpu to reduce
      the memory cost in the main gpu.
      You can check the following discussion.
      https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
      """

    def __init__(self, encz_model, encdec_model, D_model_sequence, D_model_frame,
                 criterion_recon, criterion_KL, criterion_gan,
                 x1recon_lambda=1.0, x2recon_lambda=1.0, x3recon_lambda=1.0, gan_lambda=1.0):
        super(FullModel_encdec, self).__init__()
        self.encz_model = encz_model
        self.encdec_model = encdec_model
        self.D_model_sequence = D_model_sequence
        self.D_model_frame = D_model_frame
        self.criterion_recon = criterion_recon
        self.criterion_KL = criterion_KL
        self.criterion_gan = criterion_gan
        self.x1recon_lambda = x1recon_lambda
        self.x2recon_lambda = x2recon_lambda
        self.x3recon_lambda = x3recon_lambda
        self.gan_lambda = gan_lambda

    def _anomoly_detection(self, tensor_dict):
        for n, t in tensor_dict.items():
            assert ~torch.isnan(t).any() and ~torch.isinf(t).any(), "{} got nan or inf".format(n)

    def forward(self, xt, x2t, x3t, multiplier,
                is_baseline = False, baseline_mode='VAE_NATIVE', sampling_mode='default', xt_last=None, x3t_last=None):
        assert sampling_mode in ['default', 'prior_sampling', 'momentum_sampling']
        if sampling_mode == 'momentum_sampling':
            assert xt_last is not None
            assert x3t_last is not None
        x2recon_lambda = self.x2recon_lambda
        kl_z_lambda = self.x3recon_lambda * multiplier if baseline_mode == 'VAE_ANNEAL' else self.x3recon_lambda
        if baseline_mode != 'DETERMINISTIC':
            #muvars = self.encz_model(x = torch.cat([xt_last, x3t_last], 1)) if sampling_mode == 'momentum_sampling' else self.encz_model(x = torch.cat([xt, x3t], 1))
            muvars = self.encz_model(x = torch.cat([xt, x2t, x3t] if is_baseline else [xt, x3t], 1))
            if self.encz_model.hd_z:
                mus = [muvar[:, 0 : self.encz_model.z_dim, :, :] for muvar in muvars]
                logvars = [muvar[:, self.encz_model.z_dim:, :, :] for muvar in muvars]
            else:
                mus = muvars[:, 0: self.encz_model.z_dim, :, :]
                logvars = muvars[:, self.encz_model.z_dim:, :, :]

        for l in range(1):
            if baseline_mode != 'DETERMINISTIC':
                if self.encz_model.hd_z:
                    if sampling_mode == 'prior_sampling':
                        z = [torch.randn(*mu.size(), device=mus[-1].device) for
                             mu, logvar in zip(mus, logvars)]
                    else:
                        z = [mu  + torch.exp(torch.mul(logvar, 0.5)) * torch.randn(*mu.size(), device=mus[-1].device) for
                             mu, logvar in zip(mus, logvars)]
                        self._anomoly_detection({n: t for n, t in zip(range(len(z)), z)})

                else:
                    if sampling_mode == 'prior_sampling':
                        z = torch.randn(*mus.size(), device=mus[-1].device)
                    else:
                        z = mus + torch.exp(torch.mul(logvars, 0.5)) * torch.randn(*mus.size(), device=mus.device)
                        self._anomoly_detection({'0': z})
            else:
                z = None

            xt_predict, x2t_predict, x3t_predict = self.encdec_model(x=torch.cat([xt, x2t], 1) if is_baseline else xt,
                                                                     z=z, is_baseline=is_baseline)
            self._anomoly_detection({'xt_predict': xt_predict, 'x2t_predict': x2t_predict, 'x3t_predict': x3t_predict})

            if not is_baseline:
                xt_recon_loss = self.criterion_recon(predict=xt_predict, target=xt)
                x2t_recon_loss = self.criterion_recon(predict=x2t_predict, target=x2t)
                x3t_recon_loss = self.criterion_recon(predict=x3t_predict, target=x3t)
                z_KL_loss = self.criterion_KL(mu=mus, logvar=logvars)
                x2t_gan_sequence_loss = 0.5 * self.criterion_gan(sample=self.D_model_sequence(x2t_predict), mode='real')
                x2t_gan_frame_losses = []
                for frame_idx in range(x2t.shape[1] // self.D_model_sequence.clip_length):
                    x2t_gan_frame_losses.append(0.5 * self.criterion_gan(
                        sample=self.D_model_frame(x2t_predict[:, frame_idx * 3: frame_idx * 3 + 3, :, :]), mode='real'))
                x2t_gan_frame_loss = torch.sum(torch.stack(x2t_gan_frame_losses, axis=0), axis=0)
            else:
                xt_recon_loss = 0.0
                x2t_recon_loss = self.criterion_recon(predict=x2t_predict, target=x3t)
                x3t_recon_loss = 0.0
                if baseline_mode == 'VAE_NATIVE' or baseline_mode == 'VAE_ANNEAL':
                    x2t_gan_sequence_loss = 0.0
                    x2t_gan_frame_loss = 0.0
                    z_KL_loss = self.criterion_KL(mu=mus, logvar=logvars)
                elif baseline_mode == 'DETERMINISTIC':
                    x2t_gan_sequence_loss = 0.0
                    x2t_gan_frame_loss = 0.0
                    z_KL_loss = 0.0
                elif baseline_mode == 'VAE_GAN':
                    x2t_gan_sequence_loss = 0.5 * self.criterion_gan(sample=self.D_model_sequence(x2t_predict),
                                                                     mode='real')
                    x2t_gan_frame_losses = []
                    for frame_idx in range(x2t.shape[1] // self.D_model_sequence.clip_length):
                        x2t_gan_frame_losses.append(0.5 * self.criterion_gan(
                            sample=self.D_model_frame(x2t_predict[:, frame_idx * 3: frame_idx * 3 + 3, :, :]),
                            mode='real'))
                    x2t_gan_frame_loss = torch.sum(torch.stack(x2t_gan_frame_losses, axis=0), axis=0)
                    z_KL_loss = self.criterion_KL(mu=mus, logvar=logvars)
                else:
                    raise NotImplementedError("Not implemented Baseline Mode: {}".format(baseline_mode))
            '''
            if torch.randint(0, 5, (1,)) == 0:
                x2t_recon_loss = self.criterion_recon(predict=x2t_predict, target=x2t)
            else:
                x2t_recon_loss = 0.0
            '''
        losses_all = self.x1recon_lambda * xt_recon_loss + x2recon_lambda * x2t_recon_loss + \
                     self.x3recon_lambda * x3t_recon_loss + kl_z_lambda * z_KL_loss + \
                     self.gan_lambda * (x2t_gan_sequence_loss + x2t_gan_frame_loss)

        return [torch.unsqueeze(losses_all, 0), xt_recon_loss, x2t_recon_loss, x3t_recon_loss, z_KL_loss, x2t_gan_sequence_loss, x2t_gan_frame_loss],\
               xt_predict, x2t_predict, x3t_predict


class FullToyModel_encdec(nn.Module):
    """
      Distribute the loss on multi-gpu to reduce
      the memory cost in the main gpu.
      You can check the following discussion.
      https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
      """

    def __init__(self, encz_model, encdec_model, D_model,
                 criterion_recon, criterion_KL, criterion_gan,
                 x1recon_lambda=1.0, x2recon_lambda=1.0, x3recon_lambda=1.0, gan_lambda=1.0):
        super(FullToyModel_encdec, self).__init__()
        self.encz_model = encz_model
        self.encdec_model = encdec_model
        self.D_model = D_model
        self.criterion_recon = criterion_recon
        self.criterion_KL = criterion_KL
        self.criterion_gan = criterion_gan
        self.x1recon_lambda = x1recon_lambda
        self.x2recon_lambda = x2recon_lambda
        self.x3recon_lambda = x3recon_lambda
        self.gan_lambda = gan_lambda

    def _anomoly_detection(self, tensor_dict):
        for n, t in tensor_dict.items():
            assert ~torch.isnan(t).any() and ~torch.isinf(t).any(), "{} got nan or inf".format(n)

    def forward(self, xt, x2t, x3t, multiplier, is_baseline = False, baseline_mode=None, sampling_mode='default', xt_last=None, x3t_last=None):
        assert sampling_mode in ['default', 'prior_sampling', 'momentum_sampling']
        if sampling_mode == 'momentum_sampling':
            assert xt_last is not None
            assert x3t_last is not None
        if is_baseline:
            xt = torch.cat([xt, x2t], 1)

        x2recon_lambda = self.x2recon_lambda * multiplier
        if baseline_mode != 'DETERMINISTIC':
            muvars = self.encz_model(x = torch.cat([xt_last, x3t_last], 1)) if sampling_mode == 'momentum_sampling' else self.encz_model(x = torch.cat([xt, x3t], 1))
            mus = muvars[:, 0: self.encz_model.z_dim]
            logvars = muvars[:, self.encz_model.z_dim:]

        losses_all = 0.0
        NUM_ROUND = 1
        for l in range(NUM_ROUND):
            if baseline_mode != 'DETERMINISTIC':
                if sampling_mode == 'prior_sampling':
                    z = torch.randn(*mus.size(), device=mus[-1].device)
                else:
                    z = mus + torch.exp(torch.mul(logvars, 0.5)) * torch.randn(*mus.size(), device=mus.device)
                    self._anomoly_detection({'0': z})
            else:
                z = None

            xt_predict, x2t_predict, x3t_predict = self.encdec_model(x=xt, z=z)
            self._anomoly_detection({'xt_predict': xt_predict, 'x2t_predict': x2t_predict, 'x3t_predict': x3t_predict})
            if is_baseline:
                xt_recon_loss = 0.0
                x2t_recon_loss = self.criterion_recon(predict=x2t_predict, target=x3t)
                x3t_recon_loss = 0.0
                if baseline_mode == 'VAE_NATIVE' or baseline_mode == 'VAE_ANNEAL':
                    x2t_gan_loss = 0.0
                    z_KL_loss = self.criterion_KL(mu=mus, logvar=logvars)
                elif baseline_mode == 'DETERMINISTIC':
                    x2t_gan_loss = 0.0
                    z_KL_loss = 0.0
                elif baseline_mode == 'VAE_GAN':
                    x2t_gan_loss = self.criterion_gan(sample=self.D_model(x2t_predict), mode='real')
                    z_KL_loss = self.criterion_KL(mu=mus, logvar=logvars)
                else:
                    raise NotImplementedError("Not implemented Baseline Mode: {}".format(baseline_mode))
            else:
                xt_recon_loss = self.criterion_recon(predict=xt_predict, target=xt)
                x3t_recon_loss = self.criterion_recon(predict=x3t_predict, target=x3t)
                z_KL_loss = self.criterion_KL(mu=mus, logvar=logvars)
                x2t_gan_loss = self.criterion_gan(sample=self.D_model(x2t_predict), mode='real')
                x2t_recon_loss = self.criterion_recon(predict=x2t_predict, target=x2t)

            losses_all += self.x1recon_lambda * xt_recon_loss + x2recon_lambda * x2t_recon_loss + \
                         self.x3recon_lambda * x3t_recon_loss + self.x3recon_lambda * z_KL_loss + \
                         self.gan_lambda * x2t_gan_loss
        losses_all /= NUM_ROUND

        return [torch.unsqueeze(losses_all, 0), xt_recon_loss, x2t_recon_loss, x3t_recon_loss, z_KL_loss, x2t_gan_loss, x2t_gan_loss],\
               xt_predict, x2t_predict, x3t_predict


class FullModel_D(nn.Module):
    """
      Distribute the loss on multi-gpu to reduce
      the memory cost in the main gpu.
      You can check the following discussion.
      https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
      """

    def __init__(self, D_model_sequence, D_model_frame, criterion_gan, gan_lambda=1.0):
        super(FullModel_D, self).__init__()
        self.D_model_sequence = D_model_sequence
        self.D_model_frame = D_model_frame
        self.criterion_gan = criterion_gan
        self.gan_lambda = gan_lambda

    def forward(self, x2t, x2t_predict):
        real_loss_sequence = 0.5 * self.criterion_gan(self.D_model_sequence(x2t.detach()), 'real')
        fake_loss_sequence = 0.5 * self.criterion_gan(self.D_model_sequence(x2t_predict.detach()), 'fake')

        real_losses_frame = []
        fake_losses_frame = []
        for frame_idx in range(x2t.shape[1] // self.D_model_sequence.clip_length):
            real_losses_frame.append(0.5 * self.criterion_gan(self.D_model_frame(x2t.detach()[:, frame_idx * 3: frame_idx * 3 + 3, :, :]), 'real'))
            fake_losses_frame.append(0.5 * self.criterion_gan(self.D_model_frame(x2t_predict.detach()[:, frame_idx * 3: frame_idx * 3 + 3, :, :]), 'fake'))

        real_loss_frame = torch.sum(torch.stack(real_losses_frame, axis=0), axis=0)
        fake_loss_frame = torch.sum(torch.stack(fake_losses_frame, axis=0), axis=0)

        D_losses_sequence = real_loss_sequence + fake_loss_sequence
        D_losses_frame = real_loss_frame + fake_loss_frame
        D_losses = self.gan_lambda * (D_losses_sequence + D_losses_frame)

        return [torch.unsqueeze(D_losses, 0), D_losses_sequence, D_losses_frame]


class FullToyModel_D(nn.Module):
    """
      Distribute the loss on multi-gpu to reduce
      the memory cost in the main gpu.
      You can check the following discussion.
      https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
      """

    def __init__(self, D_model, criterion_gan, gan_lambda=1.0):
        super(FullToyModel_D, self).__init__()
        self.D_model = D_model
        self.criterion_gan = criterion_gan
        self.gan_lambda = gan_lambda

    def forward(self, x2t, x2t_predict):
        real_loss = 0.5 * self.criterion_gan(self.D_model(x2t.detach()), 'real')
        fake_loss = 0.5 * self.criterion_gan(self.D_model(x2t_predict.detach()), 'fake')

        D_losses = real_loss + fake_loss

        return [torch.unsqueeze(D_losses, 0), D_losses, D_losses]


class FullModel_all(nn.Module):
    """
      Distribute the loss on multi-gpu to reduce
      the memory cost in the main gpu.
      You can check the following discussion.
      https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
      """

    def __init__(self, encz_model, encdec_model, D_model, criterion_recon, criterion_KL, criterion_gan):
        super(FullModel_all, self).__init__()
        self.encz_model = encz_model
        self.encdec_model = encdec_model
        self.D_model = D_model
        self.criterion_recon = criterion_recon
        self.criterion_KL = criterion_KL
        self.criterion_gan = criterion_gan

    def forward(self, xt, x3t, x2t, mode):
        assert mode in ['encdec', 'discriminator']
        if mode == 'encdec':
            muvars = self.encz_model(torch.cat([xt, x3t], 1))
            mus = muvars[:, 0:10, :, :]
            logvars = muvars[:, 10:, :, :]

            for l in range(1):
                #z = [mu + logvar.mul(0.5).exp_() * torch.randn(*mu.size()) for mu, logvar in zip(mus, logvars)]
                z = mus + torch.exp(torch.mul(logvars, 0.5)) * torch.randn(*mus.size(), device = mus.device)
                xt_predict, x2t_predict, x3t_predict = self.encdec_model(xt, z)
                xt_recon_loss = self.criterion_recon(xt_predict, xt)
                x3t_recon_loss = self.criterion_recon(x3t_predict, x3t)
                z_KL_loss = self.criterion_KL(mus, logvars)
                x2t_gan_loss = 0.5 * self.criterion_gan(x2t_predict, 'real')

            losses_all = xt_recon_loss + x3t_recon_loss + z_KL_loss + x2t_gan_loss

            return [torch.unsqueeze(losses_all, 0), xt_recon_loss, x3t_recon_loss, z_KL_loss, x2t_gan_loss], x2t_predict
        else:
            muvars = self.encz_model(torch.cat([xt, x3t], 1))
            mus = muvars[:, 0:10, :, :]
            logvars = muvars[:, 10:, :, :]

            #z = [mu + logvar.mul(0.5).exp_() * torch.randn(*mu.size(), device = mu.device) for mu, logvar in zip(mus, logvars)]
            z = mus + torch.exp(torch.mul(logvars, 0.5)) * torch.randn(*mus.size(), device = mus.device)
            _, x2t_predict, _ = self.encdec_model(xt, z)

            real_loss = 0.5 * self.criterion_gan(self.D_model(x2t), 'real')
            fake_loss = 0.5 * self.criterion_gan(self.D_model(x2t_predict.detach()), 'fake')

            D_losses = real_loss + fake_loss


            return [torch.unsqueeze(D_losses, 0)], x2t_predict

def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def dynamic_coeff(max_iters, cur_iters):
    import math
    #return 0.5 * (1 + math.cos(math.pi * cur_iters / max_iters))
    return math.sin((math.pi / 2) * (float(cur_iters) / float(max_iters)))
