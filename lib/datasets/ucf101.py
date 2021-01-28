import os
import logging

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

from zipfile import ZipFile


class UcfSequence(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=19,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 clip_length = 3,
                 clip_num = 3,
                 random_pos = True,
                 image_tmpl = 'image_{:05d}.jpg',
                 fixed_length = False,
                 is_baseline = False):

        super(UcfSequence, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std, )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.clip_length = clip_length
        self.clip_num = clip_num
        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        self.random_pos = random_pos
        self.fixed_length = fixed_length
        self.is_baseline = is_baseline

        self.image_tmpl = image_tmpl

        self.sequence_list = [tuple(line.split(' ')[0:2]) for line in open(list_path)]
        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]


    def read_files(self):
        files = []
        for item in self.sequence_list:
            sequence_path, length = item[0], item[1]
            name = os.path.splitext(os.path.basename(sequence_path))[0]
            files.append({
                "seq": sequence_path,
                "name": name,
                "length": int(length)
            })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def _load_image(self, idx, zip_f):
        try:
            im = Image.open(zip_f.open(self.image_tmpl.format(idx))).convert('RGB')
        except Exception as e:
            new_idx = idx - 1 if idx > 0 else idx + 1
            logging.error('Failed to open {}, open {} instead'.format(self.image_tmpl.format(idx), self.image_tmpl.format(new_idx)))
            im = Image.open(zip_f.open(self.image_tmpl.format(new_idx))).convert('RGB')

        return im

    def get(self, path, length):
        if self.fixed_length:
            length = min(length, 30)
        length = max(self.clip_length * self.clip_num, length)
        images = list()
        with ZipFile(os.path.join(self.root, path, 'RGB_frames.zip'), mode='r') as zip_f:
            sample_pos = np.random.randint(0, max(1, length - self.clip_length * self.clip_num + 1)) if self.random_pos \
            else max(0 if self.is_baseline else 3, length - self.clip_length * self.clip_num - 1)
            for p in range(sample_pos+1, sample_pos + self.clip_length * self.clip_num+1):
                seg_imgs = np.asarray(self._load_image(p, zip_f).resize((self.crop_size[1], self.crop_size[0])), dtype=np.float32)
                images.append(seg_imgs)

        return images

    def input_transform(self, sequence):
        sequence = np.concatenate(sequence, axis=-1)
        sequence = sequence / 255.0
        sequence -= self.mean * self.clip_length * self.clip_num
        sequence /= self.std * self.clip_length * self.clip_num
        return sequence

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        sequence = self.get(item['seq'], item['length'])
        sequence = np.transpose(self.input_transform(sequence), (2, 0, 1))

        sequences = [sequence[i * (self.clip_length * 3) : (i+1) * (self.clip_length * 3)].copy() for i in range(0, self.clip_num)]

        return sequences, name

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, "/home/yzzhou/workspace/code/video-prediction/lib/datasets/")
    from base_dataset import BaseDataset

    train_dataset = UcfSequence(
        root='/data/yizhou/ucf101_zip/',
        list_path='/data/yizhou/ucf101_zip/val_videofolder.txt',
        num_samples=None,
        num_classes=3,
        multi_scale=False,
        flip=False,
        base_size=512,
        crop_size=(256, 512))

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True)

    for i_iter, item in enumerate(trainloader):
        s, name = item
        for i, st in enumerate(s):
            im = Image.fromarray(np.transpose(np.uint8(st[0][0:3]), (1, 2, 0)))
            im.save('test{}.png'.format(i))
