# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch.nn as nn 
from loss.seg.ce import CrossEntropyLoss, GeneralizedCrossEntropyLoss
from loss.seg.dice import DiceLoss, MultiScaleDiceLoss
from loss.seg.dice import DiceWithCrossEntropyLoss, NoiseRobustDiceLoss
from loss.seg.exp_log import ExpLogLoss
from loss.seg.mse import MSELoss, MAELoss

SegLossDict = {'CrossEntropyLoss': CrossEntropyLoss,
    'GeneralizedCrossEntropyLoss': GeneralizedCrossEntropyLoss,
    'DiceLoss': DiceLoss,
    'MultiScaleDiceLoss': MultiScaleDiceLoss,
    'DiceWithCrossEntropyLoss': DiceWithCrossEntropyLoss,
    'NoiseRobustDiceLoss': NoiseRobustDiceLoss,
    'ExpLogLoss': ExpLogLoss,
    'MSELoss': MSELoss,
    'MAELoss': MAELoss}

