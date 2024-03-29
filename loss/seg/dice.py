# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from loss.seg.ce import CrossEntropyLoss
from loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class DiceLoss(nn.Module):
    def __init__(self, params = None):
        super(DiceLoss, self).__init__()

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        img_w   = loss_input_dict['image_weight']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        tensor_dim = len(predict.size())
        if(softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        # combien pixel weight and image weight
        if(tensor_dim == 5):
            img_w = img_w[:, None, None, None, None]
        else:
            img_w = img_w[:, None, None, None]
        pix_w = pix_w * img_w
        pix_w = reshape_tensor_to_2D(pix_w)
        tp ,fn,fp,tn = get_classwise_dice(predict, soft_y, pix_w)
        Dice=(2*tp+1e-10)/(2*tp+fn+fp+1e-10)
        acc=(tp+tn+1e-10)/(tp+fp+fn+tn+1e-10)
        sen=(tp+1e-10)/(tp+fn+1e-10)
        spec=(tn+1e-10)/(fp+tn+1e-10)
        prec=(tp+1e-10)/(tp+fp+1e-10)

        weighted_dice = Dice * cls_w
        weighted_acc = acc * cls_w
        weighted_sen = sen * cls_w
        weighted_spec = spec * cls_w
        weighted_prec = prec * cls_w
        average_dice  =  weighted_dice.sum() / cls_w.sum()
        average_acc  =  weighted_acc.sum() / cls_w.sum()
        average_sen  =  weighted_sen.sum() / cls_w.sum()
        average_spec  =  weighted_spec.sum() / cls_w.sum()
        average_prec  =  weighted_prec.sum() / cls_w.sum()
        dice_loss  = 1.0 - average_dice
        
        return dice_loss

class DiceWithCrossEntropyLoss(nn.Module):
    def __init__(self, params):
        super(DiceWithCrossEntropyLoss, self).__init__()
        self.enable_pix_weight = params['DiceWithCrossEntropyLoss_Enable_Pixel_Weight'.lower()]
        self.enable_cls_weight = params['DiceWithCrossEntropyLoss_Enable_Class_Weight'.lower()]
        self.ce_weight = params['DiceWithCrossEntropyLoss_CE_Weight'.lower()]
        dice_params = {'DiceLoss_Enable_Pixel_Weight'.lower(): self.enable_pix_weight,
                       'DiceLoss_Enable_Class_Weight'.lower(): self.enable_cls_weight}
        ce_params   = {'CrossEntropyLoss_Enable_Pixel_Weight'.lower(): self.enable_pix_weight,
                       'CrossEntropyLoss_Enable_Class_Weight'.lower(): self.enable_cls_weight}
        self.dice_loss = DiceLoss(dice_params)
        self.ce_loss   = CrossEntropyLoss(ce_params)

    def forward(self, loss_input_dict):
        loss1 = self.dice_loss(loss_input_dict)
        loss2 = self.ce_loss(loss_input_dict)
        loss = loss1 + self.ce_weight * loss2
        return loss 

class MultiScaleDiceLoss(nn.Module):
    def __init__(self, params):
        super(MultiScaleDiceLoss, self).__init__()
        self.multi_scale_weight = params['MultiScaleDiceLoss_Scale_Weight'.lower()]
        self.base_loss = DiceLoss()

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        img_w   = loss_input_dict['image_weight'] 
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']
        if(isinstance(predict, (list,tuple))):
            predict_num = len(predict)
            assert(predict_num == len(self.multi_scale_weight))
            loss   = 0.0
            weight = 0.0
            interp_mode = 'trilinear' if(len(predict[0].shape) == 5) else 'bilinear'
            for i in range(predict_num):
                soft_y_temp = nn.functional.interpolate(soft_y, 
                    size = list(predict[i].shape)[2:], mode = interp_mode)
                if(pix_w is not None):
                    pix_w_temp  = nn.functional.interpolate(pix_w, 
                        size = list(predict[i].shape)[2:], mode = interp_mode)
                else:
                    pix_w_temp = None
                temp_loss_dict = {}
                temp_loss_dict['prediction'] = predict[i]
                temp_loss_dict['ground_truth'] = soft_y_temp
                temp_loss_dict['image_weight'] = img_w 
                temp_loss_dict['pixel_weight'] = pix_w_temp
                temp_loss_dict['class_weight'] = cls_w
                temp_loss_dict['softmax'] = softmax
                temp_loss = self.base_loss(temp_loss_dict)
                loss      = loss + temp_loss * self.multi_scale_weight[i]
                weight    = weight + self.multi_scale_weight[i]
            loss = loss/weight
        else:
            loss = self.base_loss(loss_input_dict)
        return loss

class NoiseRobustDiceLoss(nn.Module):
    """
    Noise-robust Dice loss according to the following paper. 
        G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19 
        Pneumonia Lesions From CT Images, IEEE TMI, 2020.
    """
    def __init__(self, params):
        super(NoiseRobustDiceLoss, self).__init__()
        # self.enable_pix_weight = params['pixel_weight'.lower()]
        # self.enable_cls_weight = params['class_weight'.lower()]
        # self.gamma = params['NoiseRobustDiceLoss_gamma'.lower()]

    def forward(self, loss_input_dict):
        self.enable_pix_weight   = True
        self.enable_cls_weight   = True
        self.gamma=1.5
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']

        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        denominator = predict + soft_y 
        if(self.enable_pix_weight):
            if(pix_w is None):
                raise ValueError("Pixel weight is enabled but not defined")
            pix_w = reshape_tensor_to_2D(pix_w)
            numerator = numerator * pix_w
            denominator = denominator * pix_w
        numer_sum = torch.sum(numerator,  dim = 0)
        denom_sum = torch.sum(denominator,  dim = 0)
        loss_vector = numer_sum / (denom_sum + 1e-5)

        if(self.enable_cls_weight):
            if(cls_w is None):
                raise ValueError("Class weight is enabled but not defined")
            weighted_dice = loss_vector * cls_w
            loss =  weighted_dice.sum() / cls_w.sum()
        else:
            loss = torch.mean(loss_vector)   
        return loss
