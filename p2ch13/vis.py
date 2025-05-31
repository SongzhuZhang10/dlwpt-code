import matplotlib
# matplotlib.use('nbagg')

import matplotlib.pyplot as plt

from collections import namedtuple

import torch
from torch import nn as nn

import numpy as np

from p2ch13.dsets import Ct, LunaSliceSegDataset

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

clim=(-1000.0, 300)

def findPositiveSamples(start_ndx=0, limit=100):
    ds = LunaSliceSegDataset(sortby_str='label_and_size')

    positiveSample_list = []
    for sample_tup in ds.candidateInfo_list:
        if sample_tup.isNodule_bool:
            print(len(positiveSample_list), sample_tup)
            positiveSample_list.append(sample_tup)

        if len(positiveSample_list) >= limit:
            break

    return positiveSample_list

def showCandidate(series_uid, batch_ndx=None, **kwargs):
    ds = LunaSliceSegDataset(series_uid=series_uid, **kwargs)
    pos_list = [i for i, x in enumerate(ds.candidateInfo_list) if x.isNodule_bool]

    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc.index)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[int(center_irc.index)], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc.row)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:,int(center_irc.row)], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc.col)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_a[:,:,int(center_irc.col)], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc.index)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[ct_a.shape[0]//2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc.row)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:,ct_a.shape[1]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc.col)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:,:,ct_a.shape[2]//2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')


    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')
"""
SegmentationMask is NOT used for training label generation here.
It is used for data visualization only.
"""
class SegmentationMask(nn.Module):
    def __init__(self):
        super().__init__()
        """
        nn.ModuleList is a container provided by PyTorch to hold a list of nn.Module instances — like nn.Conv2d, nn.Linear, etc.
        1. Creates 7 convolutional layers, one for each radius from 1 to 7.
        2. Each layer is initialized using the _make_circle_conv(radius) method.
        3. These layers are wrapped inside an nn.ModuleList.
        Each conv layer applies a circular filter of a specific radius. Having multiple such layers lets the model perform
        morphological operations (like erosion/dilation) at different scales.
        Organs or nodules have round structures. Thus, circular filters is chosen because it
            - Respects anatomical symmetry.
            - Avoids introducing square artifacts.
            - Better mimic biological shape when doing morphological operations like erosion/dilation.
        """
        self.conv_list = nn.ModuleList([
            self._make_circle_conv(radius) for radius in range(1, 8)
        ])

    def _make_circle_conv(self, radius):
        diameter = 1 + radius * 2

        a = torch.linspace(-1, 1, steps=diameter)**2
        b = (a[None] + a[:, None])**0.5

        circle_weights = (b <= 1.0).to(torch.float32)

        conv = nn.Conv2d(1, 1, kernel_size=diameter, padding=radius, bias=False)
        conv.weight.data.fill_(1)
        conv.weight.data *= circle_weights / circle_weights.sum()

        return conv


    def erode(self, input_mask, radius, threshold=1):
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)

        #log.debug(['erode in ', radius, threshold, input_float.min().item(), input_float.mean().item(), input_float.max().item()])
        #log.debug(['erode out', radius, threshold, result.min().item(), result.mean().item(), result.max().item()])

        return result >= threshold

    def deposit(self, input_mask, radius, threshold=0):
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)

        #log.debug(['deposit in ', radius, threshold, input_float.min().item(), input_float.mean().item(), input_float.max().item()])
        #log.debug(['deposit out', radius, threshold, result.min().item(), result.mean().item(), result.max().item()])

        return result > threshold

    def fill_cavity(self, input_mask):
        cumsum = input_mask.cumsum(-1)
        filled_mask = (cumsum > 0)
        filled_mask &= (cumsum < cumsum[..., -1:])
        cumsum = input_mask.cumsum(-2)
        filled_mask &= (cumsum > 0)
        filled_mask &= (cumsum < cumsum[..., -1:, :])

        return filled_mask


    def forward(self, input_g, raw_pos_g):
        gcc_g = input_g + 1

        with torch.no_grad():
            log.info(['gcc_g', gcc_g.min(), gcc_g.mean(), gcc_g.max()])

            raw_dense_mask = gcc_g > 0.7
            dense_mask = self.deposit(raw_dense_mask, 2)
            dense_mask = self.erode(dense_mask, 6)
            dense_mask = self.deposit(dense_mask, 4)

            body_mask = self.fill_cavity(dense_mask)
            air_mask = self.deposit(body_mask & ~dense_mask, 5)
            air_mask = self.erode(air_mask, 6)

            lung_mask = self.deposit(air_mask, 5)

            raw_candidate_mask = gcc_g > 0.4
            raw_candidate_mask &= air_mask
            candidate_mask = self.erode(raw_candidate_mask, 1)
            candidate_mask = self.deposit(candidate_mask, 1)

            pos_mask = self.deposit((raw_pos_g > 0.5) & lung_mask, 2)

            neg_mask = self.deposit(candidate_mask, 1)
            neg_mask &= ~pos_mask
            neg_mask &= lung_mask

            label_g = (neg_mask | pos_mask).to(torch.float32)
            label_g = (pos_mask).to(torch.float32)
            neg_g = neg_mask.to(torch.float32)
            pos_g = pos_mask.to(torch.float32)

        mask_dict = {
            'raw_dense_mask': raw_dense_mask,
            'dense_mask': dense_mask,
            'body_mask': body_mask,
            'air_mask': air_mask,
            'raw_candidate_mask': raw_candidate_mask,
            'candidate_mask': candidate_mask,
            'lung_mask': lung_mask,
            'neg_mask': neg_mask,
            'pos_mask': pos_mask,
        }

        return label_g, neg_g, pos_g, lung_mask, mask_dict

def build2dLungMask(series_uid, center_ndx):
    mask_model = SegmentationMask().to('cuda')
    ct = Ct(series_uid)

    ct_g = torch.from_numpy(ct.hu_a[center_ndx].astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
    pos_g = torch.from_numpy(ct.positive_mask[center_ndx].astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
    input_g = ct_g / 1000

    label_g, neg_g, pos_g, lung_mask, mask_dict = mask_model(input_g, pos_g)
    mask_tup = MaskTuple(**mask_dict)

    return mask_tup
