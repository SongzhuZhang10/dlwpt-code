import math
import random

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        """
        Normalize input intensities across batches to accelerate convergence by adding a 2D BatchNorm layer
        This standardization aligns input scales across different samples, so the U-Net doesn't have to
        “relearn” how to deal with contrast differences.
        Each input channel is normalized using mean & variance computed across the batch and spatial dimensions.
        """
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs) # Instantiate the core U-Net model.
        # Compress U-Net output to the range (0, 1), representing pixel-level probabilities for binary segmentation.
        self.final = nn.Sigmoid() # Define a final activation layer.

        # Apply a unified strategy to initialize weights in convolutional and linear layers
        self._init_weights()

    def _init_weights(self):
        # Define a set of layer types eligible for custom initialization.
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                # Apply Kaiming He normal initialization to weights for layers followed by ReLU activation
                nn.init.kaiming_normal_(
                    m.weight.data,
                    mode='fan_out', # Keep gradients stable during backpropagation
                    nonlinearity='relu', # The weights will be followed by a ReLU activation.
                    a=0 # Specify the negative slope of a ReLU activation, which is 0 for standard ReLU
                )
                # Proceed only if the layer has a bias term.
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    # Determine the standard deviation range for the bias.
                    bound = 1 / math.sqrt(fan_out)
                    # Initialize bias from a normal distribution centered at 0 with small variance.
                    nn.init.normal_(m.bias, -bound, bound)


    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output

class SegmentationAugmentation(nn.Module):
    def __init__(
            self, flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

        log.info("Segmentation Augmentation Configuration:")
        log.info(f"  flip:   {self.flip   if self.flip   is not None else 'N/A'}")
        log.info(f"  offset: {self.offset if self.offset is not None else 'N/A'}")
        log.info(f"  scale:  {self.scale  if self.scale  is not None else 'N/A'}")
        log.info(f"  rotate: {self.rotate if self.rotate is not None else 'N/A'}")
        log.info(f"  noise:  {self.noise  if self.noise  is not None else 'N/A'}")


    def forward(self, input_g, label_g):
        # Create a random transformation matrix (rotation, flip, scale, offset)
        transform_t = self._build2dTransformMatrix()
        
        # Expand the tensor to match the batch size.
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        
        # Move the transform tensor/matrix to the same device as input_g (CPU or GPU).
        transform_t = transform_t.to(input_g.device, torch.float32)
        
        # Generate a sampling grid based on the 2×3 affine transform.
        affine_t = F.affine_grid(transform_t[:,:2], input_g.size(), align_corners=False)

        # Warp the input tensor using the affine grid
        # Repeat the border pixel values when sampling outside the image.
        augmented_input_g = F.grid_sample(input_g, affine_t, padding_mode='border', align_corners=False)

        # Warp the label tensor using the same affine grid
        # Note that the augmented version of the label may contain pixel values between 0 and 1 due to interpolation.
        augmented_label_g = F.grid_sample(label_g.to(torch.float32), affine_t, padding_mode='border', align_corners=False)

        if self.noise:
            # Create a new random tensor that has the same shape, same device, and same data type as the input tensor
            noise_t = torch.randn_like(augmented_input_g)
            # Set the strength of the noise by multiplying each element in noise_t by a constant.
            noise_t *= self.noise
            # Add the noise tensor element-wise to the input tensor
            augmented_input_g += noise_t

        # Return model input and ground-truth target
        # Restore the label back to a binary mask by thresholding at 0.5.
        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        # Creates a 3 × 3 identity matrix.
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                # 50% chance of flipping along x-/y-axis
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            """
            The maxtrix to be transformed by the this transform matrix should conform 
            to row vector convention. This means that only the bottom row stores the
            translation.
            """
            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                # Translation is in the bottom row.
                transform_t[2,i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if self.rotate:
            # Takes a random angle in radians, so in the range 0 .. 2{pi}
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            # Rotation matrix for the 2D rotation by the random angle
            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            # Apply the rotation to the transformation matrix using matrix multiplication
            transform_t @= rotation_t

        return transform_t