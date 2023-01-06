"""
SYNOPSIS
    Implementation of multilayer perceptron network using JAX libraries
DESCRIPTION

    Contains two modules:
    a) TrajDataset class - to load the dataset
    b) MLP - defines the layers and depth of the multilayer perceptron
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

import flax
from flax import linen as nn
import torch.utils.data as data

## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')  # For export
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
print("Using jax", jax.__version__)


import torch
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    """
    Dataset class inherited from torch modules
    """
    def __init__(self, xtraj, utraj, rtraj, xtraj_):
        """
        Input:
            - xtraj:        np.array(N, p), sequence of states
            - utraj:        np.array(N, q), sequence of inputs
            - rtraj:        np.array(N,), sequence of rewards
            - xtraj_:       np.array(N, p), sequence of next states
        """
        self.xtraj = xtraj
        self.utraj = utraj
        self.rtraj = rtraj
        self.xtraj_ = xtraj_

    def __len__(self):
        return len(self.xtraj)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.xtraj[idx], self.utraj[idx], self.rtraj[idx], self.xtraj_[idx]


class MLP(nn.Module):
    num_hidden: list
    num_outputs: int

    def setup(self):
        self.linear = [nn.Dense(features=self.num_hidden[i]) for i in range(len(self.num_hidden))]
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        for i in range(len(self.num_hidden)):
            x = self.linear[i](x)
            x = nn.gelu(x)
        x = self.linear2(x)
        return x