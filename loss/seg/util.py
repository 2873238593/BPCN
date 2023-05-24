# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
def get_soft_label(input_tensor, num_class, data_type = 'float'):
    """
        convert a label tensor to one-hot label 
        input_tensor: tensor with shae [B, 1, D, H, W] or [B, 1, H, W]
        output_tensor: shape [B, num_class, D, H, W] or [B, num_class, H, W]
    """

    shape = input_tensor.shape
    if len(shape) == 5:
        output_tensor = torch.nn.functional.one_hot(input_tensor[:, 0], num_classes = num_class).permute(0, 4, 1, 2, 3)
    elif len(shape) == 4:
        output_tensor = torch.nn.functional.one_hot(input_tensor[:, 0], num_classes = num_class).permute(0, 3, 1, 2)
    else:
        raise ValueError("dimention of data can only be 4 or 5: {0:}".format(len(shape)))
    
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class  = list(x.size())[1]
    if(tensor_dim == 5):
        x_perm  = x.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        x_perm  = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    y = torch.reshape(x_perm, (-1, num_class)) 
    return y 

def reshape_prediction_and_ground_truth(predict, soft_y):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    predict = torch.reshape(predict, (-1, num_class)) 
    soft_y  = torch.reshape(soft_y,  (-1, num_class))
      
    return predict, soft_y

def get_classwise_dice(predict, soft_y, pix_w = None):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """
    
    if(pix_w is None):
        zong=torch.full(soft_y.size(),1).to('cuda:0')
        zong=torch.sum(zong,dim=0)
        y_vol = torch.sum(soft_y,  dim = 0)
       
        p_vol = torch.sum(predict, dim = 0)
        intersect = torch.sum(soft_y * predict, dim = 0)
    else:
        y_vol = torch.sum(soft_y * pix_w,  dim = 0)
        p_vol = torch.sum(predict * pix_w, dim = 0)
        intersect = torch.sum(soft_y * predict * pix_w, dim = 0)
        zong=(torch.full(soft_y.size(),1).to('cuda:0')* pix_w)
        zong=torch.sum(zong,dim=0)
    tp=intersect[1]
    fn=(p_vol-intersect)[1]
    fp=(y_vol-intersect)[1]
    tn=(zong-p_vol-y_vol+intersect)[1]

    dice_score = (2.0 * intersect + 1e-10)/ (y_vol + p_vol + 1e-10)
    return tp ,fn,fp,tn

# def get_classwise_dice(predict, soft_y, pix_w = None):
#     """
#     get dice scores for each class in predict (after softmax) and soft_y
#     """
    
#     # if(pix_w is None):
#     if (pix_w is None):
#         zong=torch.full(soft_y.size(),1).to('cuda:0')
#         zong=torch.sum(zong,dim=0)
        
#         y_vol = torch.sum(soft_y,  dim = 0)
       
#         # p_vol = torch.sum(predict, dim = 0)
#         # intersect = torch.sum(soft_y * predict, dim = 0)
#         # zong[1]=0
#         tmp=torch.from_numpy((predict.cpu().detach().numpy()>0.5))
#         device=torch.device('cuda:0')
#         tmp=tmp.to(device)
#         # tmp=Variable(tmp,requires_grad=True)
#         intersect = torch.sum(soft_y* tmp, dim = 0)
#         p_vol = torch.sum(tmp, dim = 0)
        
#     else:
#         y_vol = torch.sum(soft_y * pix_w,  dim = 0)
#         tmp=torch.from_numpy((predict.cpu().detach().numpy()>0.5))
#         device=torch.device('cuda:0')
#         tmp=tmp.to(device)
#         tmp=Variable(tmp,requires_grad=True)
#         p_vol = torch.sum(tmp * pix_w, dim = 0)
#         intersect = torch.sum(soft_y * tmp * pix_w, dim = 0)
#         zong=torch.full((soft_y).size(),1).to('cuda:0')
#         zong=torch.sum(zong* pix_w,dim=0)
#         # zong[0]=0
#     # print('============================')
#     # print(zong)
#     tp=intersect
#     # print(tp)
#     fn=p_vol-intersect
#     # print(fn)
#     fp=y_vol-intersect
#     # print(fp)
#     tn=zong-p_vol-y_vol+intersect
#     # print(tn)
#     # print('============================')
#     #TP=area_intersection
#     # FN=area_output-area_intersection
#     # FP=area_target-area_intersection
#     # TN=zong-area_union
#     dice_score = (2.0 * intersect + 1e-10)/ (y_vol + p_vol + 1e-10)

#     return tp ,fn,fp,tn