import torch
import torch.nn as nn
import lightningaddon as la


def conv_layer(shape, ks, stride, padding):
    return nn.Sequential(
        nn.Conv2d(shape, shape, ks, stride, padding),
        nn.BatchNorm2d(shape),
        nn.ReLU(inplace=True),
    )


def conv_separable(shape, stride, padding):
    part1 = conv_layer(shape, 3, stride, padding)
    part2 = conv_layer(shape, 1, stride, padding)
