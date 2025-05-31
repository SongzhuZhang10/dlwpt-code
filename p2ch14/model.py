import math
import random
from collections import namedtuple
import numpy as np

import torch
from torch import nn as nn
from typing import Tuple
import torch.nn.functional as F

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def augment3d(volume: torch.Tensor) -> torch.Tensor:
    """
    Perform random spatial augmentation on a 3D volume tensor (N, C, D, H, W).
    The augmentation operations (identical to the original implementation) include:
        1. Random flipping along each axis with 50% probability;
        2. Random translation within ±0.1 units along each axis
           (translations are written into transform[3, axis], preserving original behavior);
        3. Random rotation around the Z-axis in the range [0, 2π).

    Parameters
    ----------
    volume : torch.Tensor
        Input tensor with shape (N, C, D, H, W). Output will be on the same device.

    Returns
    -------
    torch.Tensor
        Augmented tensor with the same shape as *volume*.
    """
    device = volume.device
    dtype = torch.float32

    # ---------- Initialize 4×4 homogeneous transformation matrix ----------
    transform = torch.eye(4, dtype=dtype, device=device)

    # ---------- Flipping and translation ----------
    max_translation = 0.1
    for axis in range(3):
        # Random flipping
        if random.random() > 0.5:
            transform[axis, axis] *= -1

        # Random translation (written to row 4 instead of column 4, as in the original)
        transform[3, axis] = max_translation * (2 * random.random() - 1)

    # ---------- Random rotation around the Z-axis ----------
    angle = random.random() * 2 * math.pi
    sin_a, cos_a = math.sin(angle), math.cos(angle)

    rotation = torch.tensor(
        [
            [cos_a, -sin_a, 0, 0],
            [sin_a,  cos_a, 0, 0],
            [0,      0,     1, 0],
            [0,      0,     0, 1],
        ],
        dtype=dtype,
        device=device,
    )

    transform @= rotation

    # ---------- Generate affine grid and resample ----------
    affine = F.affine_grid(
        transform[:3].unsqueeze(0).expand(volume.size(0), -1, -1),
        size=volume.shape,
        align_corners=False,
    )

    augmented = F.grid_sample(
        volume,
        affine,
        padding_mode="border",
        align_corners=False,
    )

    return augmented


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        
        # Create a 3D batch normalization layer that is applied to the input
        # tensor before it enters the convolutional blocks.
        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        """
        Number of input features is determined by shape of input tensor
        which by definition is of shape [1, 32, 48, 48].
        At the output of block 4, the shape becomes [64, 2, 3, 3].
        64 * 2 * 3 * 3 = 1152
        """
        self.head_linear = nn.Linear(in_features=1152, out_features=2)

        # Turn each row into a probability distribution over classes
        self.head_activation = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self) -> None:

        # Layer types that require initialization
        _target_layers = (
            nn.Linear,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        )

        for layer in self.modules():
            if type(layer) in _target_layers:
                # Avoid recording operations in the computational graph
                with torch.no_grad():
                    # --- Weight initialization ---
                    nn.init.kaiming_normal_(
                        layer.weight,
                        a=0,
                        mode="fan_out",
                        nonlinearity="relu",
                    )

                    # --- Bias initialization ---
                    if layer.bias is not None:
                        _, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        std_bound = 1.0 / math.sqrt(fan_out)
                        nn.init.normal_(layer.bias, mean=-std_bound, std=std_bound)


    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        # Reshape the tensor in-place from (N, C, D, H, W) to (N, CxDxHxW)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.head_linear(conv_flat)

        # Return logit and probability
        return linear_output, self.head_activation(linear_output)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)


    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')


class SegmentationMask(nn.Module):
    """
    Provides basic morphological operations using fixed convolution kernels.

    Public Methods
    --------------
    erode(mask, radius, threshold=1)   →  torch.BoolTensor
        Erosion: keeps a pixel only if the circular neighborhood contains at least `threshold` foreground pixels.

    deposit(mask, radius, threshold=0) →  torch.BoolTensor
        Dilation: sets a pixel as foreground if the circular neighborhood contains more than `threshold` foreground pixels.

    fill_cavity(mask)                  →  torch.BoolTensor
        Fills holes that are fully surrounded by foreground pixels (in both horizontal and vertical directions).
    """

    _MAX_RADIUS: int = 7  # Pre-build convolution kernels for radius 1 to 7

    def __init__(self) -> None:
        super().__init__()

        # Precompute circular convolution filters
        self._conv_list = nn.ModuleList(
            [self._build_circular_conv(r) for r in range(1, self._MAX_RADIUS + 1)]
        )

    # ------------------------------------------------------------------ #
    # Internal Utilities                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_circular_conv(radius: int) -> nn.Conv2d:
        """
        Constructs a 1x1 channel nn.Conv2d layer with weights shaped like a circle of given radius.
        All weights inside the circle are 1, outside are 0, normalized so their sum is 1.

        When applied to a {0,1} mask, the output represents the count of foreground pixels in the neighborhood.
        """
        diameter = 2 * radius + 1

        # Create a circular mask with specified diameter
        axis = torch.linspace(-1.0, 1.0, steps=diameter) ** 2
        distance = (axis[None] + axis[:, None]).sqrt()
        kernel = (distance <= 1.0).float()  # Inside circle = 1, outside = 0

        conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=diameter,
            padding=radius,
            bias=False,
        )

        with torch.no_grad():
            conv.weight.fill_(1.0)
            conv.weight.mul_(kernel / kernel.sum())  # Normalize weights

        return conv

    # ------------------------------------------------------------------ #
    # Public Morphological Operations                                   #
    # ------------------------------------------------------------------ #
    def erode(
        self, mask: torch.Tensor, radius: int, threshold: int = 1
    ) -> torch.Tensor:
        """
        Erosion: keeps only those pixels whose circular neighborhood contains
        at least `threshold` number of foreground pixels.
        """
        conv = self._conv_list[radius - 1]
        neighbour_sum = conv(mask.float())
        return neighbour_sum >= threshold

    def deposit(
        self, mask: torch.Tensor, radius: int, threshold: int = 0
    ) -> torch.Tensor:
        """
        Dilation: sets pixels as foreground if their circular neighborhood
        contains more than `threshold` number of foreground pixels.
        """
        conv = self._conv_list[radius - 1]
        neighbour_sum = conv(mask.float())
        return neighbour_sum > threshold

    @staticmethod
    def fill_cavity(mask: torch.Tensor) -> torch.Tensor:
        """
        Fills regions that are completely enclosed by foreground pixels
        in both horizontal and vertical directions.
        """
        # Horizontal cumulative sum
        cumsum_x = mask.cumsum(dim=-1)
        filled = (cumsum_x > 0) & (cumsum_x < cumsum_x[..., -1:])

        # Vertical cumulative sum
        cumsum_y = mask.cumsum(dim=-2)
        filled &= (cumsum_y > 0) & (cumsum_y < cumsum_y[..., -1:, :])

        return filled
