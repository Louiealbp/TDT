import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

class ImpalaResBlock(nn.Module):

    def __init__(self, depth):
        super().__init__()
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding = 1)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(depth, depth, 3, padding = 1)

    def forward(self, inputs):
        out = self.act1(inputs)
        out = self.conv1(out)
        out = self.act2(out)
        out = self.conv2(out)
        return out + inputs

class ImpalaConvSequence(nn.Module):

    def __init__(self, depth, n_input_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_input_channels, depth, 3, padding = 1)
        self.act = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size = [3, 3], stride = 2)
        self.res_block1 = ImpalaResBlock(depth)
        self.res_block2 = ImpalaResBlock(depth)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.act(out)
        out = self.pooling(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        return out

class ImpalaCNN(nn.Module):

    def __init__(self, embd_dim, n_input_channels, arch = [16, 32, 32]):
        super().__init__()
        self.blocks = [ImpalaConvSequence(x, y) for x, y in zip(arch, [n_input_channels] + arch[:-1])]
        self.blocks = nn.Sequential(*self.blocks)
        self.flatten = nn.Flatten()
        self.act = nn.ReLU()
        self.dense = nn.Linear(2592, embd_dim)

    def forward(self, inputs):
        out = inputs.float()
        out = self.blocks(out)
        out = self.flatten(out)
        out = self.act(out)
        # print(out.shape)
        out = self.dense(out)
        return out
