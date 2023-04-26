# -*- coding: utf-8 -*-
from __future__ import print_function, division
from transform.gamma_correction import  ChannelWiseGammaCorrection
from transform.gray2rgb import GrayscaleToRGB
from transform.flip import RandomFlip
from transform.pad import Pad
from transform.rotate import RandomRotate
from transform.rescale import Rescale, RandomRescale  
from transform.threshold import * 
from transform.normalize import *
from transform.crop import *
from transform.label_convert import * 

TransformDict = {
    'ChannelWiseGammaCorrection': ChannelWiseGammaCorrection,
    'ChannelWiseThreshold': ChannelWiseThreshold,
    'ChannelWiseThresholdWithNormalize': ChannelWiseThresholdWithNormalize,
    'CropWithBoundingBox': CropWithBoundingBox,
    'CenterCrop': CenterCrop,
    'GrayscaleToRGB': GrayscaleToRGB,
    'LabelConvert': LabelConvert,
    'LabelConvertNonzero': LabelConvertNonzero,
    'LabelToProbability': LabelToProbability,
    'NormalizeWithMeanStd': NormalizeWithMeanStd,
    'NormalizeWithMinMax': NormalizeWithMinMax,
    'NormalizeWithPercentiles': NormalizeWithPercentiles,
    'RandomCrop': RandomCrop,
    'RandomResizedCrop': RandomResizedCrop,
    'RandomRescale': RandomRescale,
    'RandomFlip': RandomFlip,
    'RandomRotate': RandomRotate,
    'ReduceLabelDim': ReduceLabelDim,
    'Rescale': Rescale,
    'Pad': Pad,
}
