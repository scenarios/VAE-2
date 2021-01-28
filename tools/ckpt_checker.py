import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of vp")
parser.add_argument('--ckptpth', type=str, default='/data/yizhou/log/vp/output/toyexample/toyexample_1_1_1_z_2//checkpoint_encdec.pth.tar')
parser.add_argument('--ckptpth_1', type=str, default='/home/yzzhou/workspace/code/video-prediction/experiments/cityscapes/')
parser.add_argument('--ckptpth_2', type=str, default='/home/yzzhou/workspace/code/video-prediction/experiments/cityscapes/')

import os
import numpy as np
import csv

def alpha_checker(state_dict, path=''):
    with open(os.path.join(path, 'p_log.csv'), mode='w') as csv_file:
        fields = ['index', 'S', 'T']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()

        records = {}
        for name, value in state_dict.items():
            if 'p_logit' in name and 'classifier' not in name:
                print('{} {}'.format(name, value.sigmoid().item()))
                name = name.replace('module.features.', '')
                name = name.replace('.conv1.p_logit', '')
                name = name.replace('.conv.p_logit', '')
                name = name.replace('bottleneck', 'S')
                name = name.replace('temporal', 'T')
                name = name.replace('denseblock', 'B')
                name = name.replace('denselayer', 'L')
                name = name.replace('original', 'S')
                if '.S' in name:
                    if name.replace('.S', '') in records.keys():
                        records[name.replace('.S', '')]['S'] = value.sigmoid().item()
                    else:
                        records[name.replace('.S', '')] = {}
                        records[name.replace('.S', '')]['S'] = value.sigmoid().item()
                    #csv_writer.writerow({'index': name, 'S': value.sigmoid().item(), 'T': state_dict[name.replace('.S', '.T')].sigmoid().item()})
                elif '.T' in name:
                    if name.replace('.T', '') in records.keys():
                        records[name.replace('.T', '')]['T'] = value.sigmoid().item()
                    else:
                        records[name.replace('.T', '')] = {}
                        records[name.replace('.T', '')]['T'] = value.sigmoid().item()
                else:
                    pass

        for name, value in records.items():
            csv_writer.writerow({'index': name, 'S': value['S'], 'T': value['T']})


def sptp_checker(state_dict):
    t_count = 0
    s_count = 0
    for name, value in state_dict.items():
        if 'p_logit' in name and 'classifier' not in name:
            #print('{}: {}'.format(name, value.sigmoid()))
            stensor = 1- np.floor(state_dict[name.replace('p_logit', 'unif_noise_variable')].cpu().item() + value.sigmoid().cpu().item())
            if int(stensor) == 0:
                if 'temporal' in name:
                    t_count += 1
                elif 'bottleneck' or 'original' in name:
                    s_count += 1
                print('{}: {}'.format(name, stensor))
        #if 'norm' in name:
        #    print('{}: {}'.format(name, value))
    print('{}: {}'.format('t_count', t_count))
    print('{}: {}'.format('s_count', s_count))


def print_param(state_dict):
    for name, value in state_dict.items():
        print(name)


def param_comp(state_dict_1, state_dict_2):
    import torch
    for name, value in state_dict_1.items():
        if name in state_dict_2.keys():
            print(name)
            error = torch.sum(value - state_dict_2[name])
            assert error == 0, "error ={}".format.str(error)


def param_rename(state_dict):
    names = []
    for name, value in state_dict.items():
        if 'encdec_mode.' in name:
            print(name)
            names.append(name)

    for name in names:
        v = state_dict[name]
        del state_dict[name]
        new_name = name.replace('encdec_mode.', 'encdec_model.')
        state_dict[new_name] = v

    return state_dict



def main():
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.ckptpth, map_location='cpu')
    checkpoint['state_dict'] = param_rename(checkpoint['state_dict'])
    torch.save(checkpoint, args.ckptpth)


if __name__ == '__main__':
    main()