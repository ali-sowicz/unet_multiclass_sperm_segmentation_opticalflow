# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import math
from typing import Tuple

import numpy as np
import torch


try:
    import cv2
except ImportError:
    _HAS_CV2 = False
else:
    _HAS_CV2 = True


def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)  #tensor([0., 7.])
    indices = torch.clamp(indices, 0, t - 1).long() #tensor([0,7])
    return torch.index_select(x, temporal_dim, indices)



def uniform_temporal_subsample_repeated(
    frames: torch.Tensor, frame_ratios: Tuple[int], temporal_dim: int = -3
) -> Tuple[torch.Tensor]:
    """
    Prepare output as a list of tensors subsampled from the input frames. Each tensor
        maintain a unique copy of subsampled frames, which corresponds to a unique
        pathway.

    Args:
        frames (tensor): frames of images sampled from the video. Expected to have
            torch tensor (including int, long, float, complex, etc) with dimension
            larger than one.
        frame_ratios (tuple): ratio to perform temporal down-sampling for each pathways.
        temporal_dim (int): dimension of temporal.

    Returns:
        frame_list (tuple): list of tensors as output.
    """
    temporal_length = frames.shape[temporal_dim]
    frame_list = []
    for ratio in frame_ratios:
        pathway = uniform_temporal_subsample(
            frames, temporal_length // ratio, temporal_dim
        )
        frame_list.append(pathway)

    return frame_list

