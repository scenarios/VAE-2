from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

HID_DIM = 128
Z_DIM = 8
INPUT_DIM = 10
logger = logging.getLogger(__name__)

class toy_fc(nn.Module):
    def __init__(self, config):
        super(toy_fc, self).__init__()
        self.is_baseline = config.MODEL.EXTRA.IS_BASELINE
        self.baseline_mode = config.MODEL.EXTRA.BASELINE_MODE

        self.I_e_dim = INPUT_DIM * 2 if self.is_baseline else INPUT_DIM
        self.I_s_dim = INPUT_DIM
        self.v_dim = INPUT_DIM
        self.z_dim = 0 if self.baseline_mode == 'DETERMINISTIC' else Z_DIM

        self.h1 = nn.Sequential(
            nn.Linear(self.I_e_dim, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.h2 = nn.Sequential(
            nn.Linear(HID_DIM, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(HID_DIM, self.v_dim)

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _gen_code_map(self, feature, code=None):

        return torch.randn(feature.shape[0], self.z_dim, device=feature[-1].device).detach() \
            if code is None else code

    def forward(self, x, *args, **kwargs):
        x = self.h2(self.h1(x))

        return self.output(x)


class toy_fc_EDz(toy_fc):
    def __init__(self, config):
        super(toy_fc_EDz, self).__init__(config)
        self.h1 = nn.Sequential(
            nn.Linear(self.I_e_dim + self.v_dim, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(HID_DIM, Z_DIM * 2)


class toy_fc_ED(toy_fc):
    def __init__(self, config):
        super(toy_fc_ED, self).__init__(config)
        self.h1 = nn.Sequential(
            nn.Linear(self.I_e_dim + self.z_dim if self.is_baseline else self.I_e_dim + 2 * self.z_dim, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.decp_h1 = nn.Sequential(
            nn.Linear(self.I_s_dim + self.z_dim, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.decp_h2 = nn.Sequential(
            nn.Linear(HID_DIM, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.decp_output = nn.Linear(HID_DIM, self.I_e_dim)

        self.decf_h1 = nn.Sequential(
            nn.Linear(self.I_s_dim + self.z_dim, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.decf_h2 = nn.Sequential(
            nn.Linear(HID_DIM, HID_DIM),
            #nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.decf_output = nn.Linear(HID_DIM, self.v_dim)

    def _encoder_forward(self, x, z):
        codemaps_z = self._gen_code_map(x, z)
        codemaps_r = self._gen_code_map(x)
        if self.is_baseline:
            if self.baseline_mode == 'DETERMINISTIC':
                x = self.h2(self.h1(x))
            else:
                x = self.h2(self.h1(torch.cat([x, codemaps_z], -1)))
        else:
            x = self.h2(self.h1(torch.cat([x, codemaps_z, codemaps_r], -1)))

        return self.output(x)

    def _decoder_future_forward(self, x, z):
        codemaps_z = self._gen_code_map(x, z)
        if self.is_baseline and self.baseline_mode == 'DETERMINISTIC':
            x = self.decf_h2(self.decf_h1(x))
        else:
            x = self.decf_h2(self.decf_h1(torch.cat([x, codemaps_z], -1)))

        return self.decf_output(x)

    def _decoder_past_forward(self, x, z):
        codemaps_z = self._gen_code_map(x, z)
        if self.is_baseline and self.baseline_mode == 'DETERMINISTIC':
            x = self.decp_h2(self.decp_h1(x))
        else:
            x = self.decp_h2(self.decp_h1(torch.cat([x, codemaps_z], -1)))

        return self.decp_output(x)

    def forward(self, x, z=None, *args, **kwargs):
        x2 = self._encoder_forward(x, z)
        x1 = self._decoder_past_forward(x2, z)
        x3 = self._decoder_future_forward(x2, z)

        return x1, x2, x3


class toy_fc_Dsc(toy_fc):
    def __init__(self, config):
        super(toy_fc_Dsc, self).__init__(config)
        self.h1 = nn.Sequential(
            nn.Linear(self.I_s_dim, HID_DIM),
            # nn.BatchNorm1d(HID_DIM, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(HID_DIM, 1)


def get_encdec_model(config):
    model = toy_fc_ED(config)
    model.init_weights()

    return model


def get_encz_model(config):
    model = toy_fc_EDz(config)
    model.init_weights()

    return model


def get_D_model(config):
    model = toy_fc_Dsc(config)
    model.init_weights()

    return model


