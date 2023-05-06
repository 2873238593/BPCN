# -*- coding: utf-8 -*-
"""
An implementation of the U-Net paper:
    Olaf Ronneberger, Philipp Fischer, Thomas Brox:
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    MICCAI (3) 2015: 234-241
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
"""
from __future__ import print_function, division
from typing import ForwardRef

import torch
import torch.nn as nn
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from util.visualize import VisualizeFeatureMapPCA
import cv2
# import tensorflow as tf

from torch.nn.modules.activation import ReLU

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)


class DownBlockmax(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlockmax, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DownBlockavg(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlockavg, self).__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class SELayer(nn.Module):
    def __init__(self,channel,reduction=2):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
        self.fs=nn.Sequential(
            # nn.Linear(12*1*128*128,12*1*16*16,bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(12*1*16*16,12*1*128*128,bias=False),
            nn.Sigmoid()
        )
        self.conv=nn.Conv2d(2,1,kernel_size=7,stride=1,padding=3)
    def forward(self,x):
        b,c,h,w=x.size()
        y_1=self.avg_pool(x).view(b,c)
        y_2=self.max_pool(x).view(b,c)
        y=y_1*y_2
        y=self.fc(y).view(b,c,1,1)
        y=x*y.expand_as(x)

        x_avg=torch.mean(x,dim=1,keepdim=True)
        x_max,_=torch.max(x,dim=1,keepdim=True)
        x_output=torch.cat([x_avg,x_max],dim=1)
        x_output=self.conv(x_output)
        x_output=self.fs(x_output)

        return x_output.expand_as(x)*y
    
    def corrd_grid(batch,h,w):
        corrds=torch.meshgrid(torch.arange(h),torch.arange(w))
        corrds=torch.stack(corrds[::-1],dim=0).float()
        return corrds[None].repeat(batch,1,1,1)



class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        """
        in_channels1: channel of high-level features
        in_channels2: channel of low-level features
        out_channels: output channel number
        dropout_p: probability of dropout
        """
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PAFNet(nn.Module):
    def __init__(self, params):
        super(PAFNet, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1_0  = DownBlockmax(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2_0  = DownBlockmax(self.ft_chns[1]*2, self.ft_chns[2], self.dropout[2])
        self.down3_0  = DownBlockmax(self.ft_chns[2]*2, self.ft_chns[3], self.dropout[3])
        self.down4_0  = DownBlockmax(self.ft_chns[3]*2, self.ft_chns[4], self.dropout[4])

        self.down1_1  = DownBlockavg(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2_1  = DownBlockavg(self.ft_chns[1]*2, self.ft_chns[2], self.dropout[2])
        self.down3_1  = DownBlockavg(self.ft_chns[2]*2, self.ft_chns[3], self.dropout[3])
        self.down4_1  = DownBlockavg(self.ft_chns[3]*2, self.ft_chns[4], self.dropout[4])

        self.se1=SELayer(self.ft_chns[1]*2,reduction=4)
        self.se2=SELayer(self.ft_chns[2]*2,reduction=8)
        self.se3=SELayer(self.ft_chns[3]*2,reduction=16)
        self.se4=SELayer(self.ft_chns[4]*2,reduction=32)
        self.se5=SELayer(self.ft_chns[3],reduction=16)

        self.squeeze_body_edge1 = SqueezeBodyEdge(self.ft_chns[1]*2, nn.BatchNorm2d)
        self.squeeze_body_edge2 = SqueezeBodyEdge(self.ft_chns[2]*2, nn.BatchNorm2d)
        self.squeeze_body_edge3 = SqueezeBodyEdge(self.ft_chns[3]*2, nn.BatchNorm2d)
        self.squeeze_body_edge4 = SqueezeBodyEdge(self.ft_chns[4]*2, nn.BatchNorm2d)
        self.squeeze_body_edge = SqueezeBodyEdge(self.ft_chns[3], nn.BatchNorm2d)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 0.0, self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 0.0, self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 0.0, self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 0.0, self.bilinear) 

        self.upsample1 = BasicConv2d(self.ft_chns[4], self.ft_chns[3], 3, padding=1)
        self.upsample2 = BasicConv2d(self.ft_chns[2], self.ft_chns[3], 3, padding=1)
        self.upsample3 = BasicConv2d(self.ft_chns[3], self.ft_chns[4], 3, padding=1)
        self.upsample4 = BasicConv2d(64, 512, 3, padding=1)
        self.upsample5 = BasicConv2d(64, 512, 3, padding=1)

        # self.upsample_edge1 = BasicConv2d(self.ft_chns[2]*3, self.ft_chns[2]*2, 3, padding=1)
        self.upsample_edge2 = BasicConv2d(self.ft_chns[3]*3, self.ft_chns[3]*2, 3, padding=1)
        self.upsample_edge3 = BasicConv2d(self.ft_chns[4]*3, self.ft_chns[3], 3, padding=1)
        # self.upsample3 = BasicConv2d(64, 32, 3, padding=1)
        # self.conv_upsample3 = BasicConv2d(2048, 1024, 3, padding=1)
        # self.conv_upsample4 = BasicConv2d(4096, 2048, 3, padding=1)
        # self.conv_upsample5 = BasicConv2d(4096, edge_dim, 3, padding=1)
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,  
            kernel_size = 3, padding = 1)
        self.out = nn.Sequential(
            nn.Conv2d(512, self.n_class, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_class),
            nn.ReLU(),
            # nn.Conv2d(2, 2, kernel_size=1),
            # nn.BatchNorm2d(2),
            # nn.ReLU(),
            # nn.Dropout(0.1),
        )
        edge_dim=64
        end_dim=4
        self.bot_aspp = nn.Sequential(
            nn.Conv2d(512, edge_dim,  kernel_size=3,padding=1, bias=False), 
            nn.BatchNorm2d(edge_dim),
            nn.ReLU(inplace=True),
        )

        self.bot_fine = nn.Conv2d(32, end_dim, kernel_size=3,padding=1, bias=False)



        self.edge_fusion = nn.Conv2d(edge_dim + end_dim, edge_dim,1,bias=False)
        self.sigmoid_edge = nn.Sigmoid()

        self.edge_out = nn.Sequential(
            nn.Conv2d(edge_dim, end_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(end_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(end_dim, 1, kernel_size=1, bias=False)
        )
        self.final_seg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            )
        

        self.x4sample=nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.relu=nn.ReLU(inplace=True)
        self.number=0

    def forward(self, x):


        x0 = self.in_conv(x)
        x1_max = self.down1_0(x0)
        x1_avg = self.down1_1(x0)
        x1=self.se1(torch.cat([x1_max,x1_avg],dim=1))
        x2_max = self.down2_0(x1)
        x2_avg = self.down2_1(x1)
        x2=self.se2(torch.cat([x2_max,x2_avg],dim=1))
        x3_max = self.down3_0(x2)
        x3_avg = self.down3_1(x2)
        x3=self.se3(torch.cat([x3_max,x3_avg],dim=1))
        x4_max = self.down4_0(x3)
        x4_avg = self.down4_1(x3)
        x4=self.se4(torch.cat([x4_max,x4_avg],dim=1))

        size=64
        x3=F.interpolate(x3,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_3=self.upsample1(x3)
        x1_1=x_3
        x2_1=x_3*x2
        # x3=F.interpolate(x3,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x2=F.interpolate(x2,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x1=F.interpolate(x1,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x3_1=x_3*x2*self.upsample2(x1)
        x1_1=F.interpolate(x1_1,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x2_2=torch.cat([x1_1,x2_1],dim=1)
        x2_2=F.interpolate(x2_2,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x3_2=torch.cat([self.upsample3(x3_1),x2_2],dim=1)
        x4=F.interpolate(x4,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        aspp=torch.cat([x3_2,x4],dim=1)
        aspp1=aspp

        _,x_edge_1=self.squeeze_body_edge1(x1)
        _,x_edge_2=self.squeeze_body_edge2(x2)
        _,x_edge_3=self.squeeze_body_edge3(x3)
        _,x_edge_4=self.squeeze_body_edge4(x4)

        x_edge_3=F.interpolate(x_edge_3,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_edge_3=self.upsample1(x_edge_3)
        x1_1=x_edge_3
        x2_1=x_edge_3*x_edge_2
        # x3=F.interpolate(x3,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_edge_2=F.interpolate(x_edge_2,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_edge_1=F.interpolate(x_edge_1,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_edge_3_1=x_edge_3*x_edge_2*self.upsample2(x_edge_1)
        x1_1=F.interpolate(x1_1,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x2_2=torch.cat([x1_1,x2_1],dim=1)
        x2_2=F.interpolate(x2_2,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x3_2=torch.cat([self.upsample3(x_edge_3_1),x2_2],dim=1)
        x_edge_4=F.interpolate(x_edge_4,size=torch.Size([size,size]),mode='bilinear',align_corners=True)
        x_edge=torch.cat([x3_2,x_edge_4],dim=1)
        aspp=aspp+x_edge
        # x_edge = self.bot_aspp(x_edge)



        aspp = self.bot_aspp(aspp)
        x_1=x1
        m2=x1
        seg_body, seg_edge = self.squeeze_body_edge(aspp,x_1)
        # for i in range(x.shape[0]):
        #     img2=VisualizeFeatureMapPCA(seg_edge[i])
        #     cv2.imwrite('result/'+str(name[i])+'_edge.png',img2)
        # aspp=self.bot_aspp1(aspp1)
        aspp=aspp1
        _,_,w,h=aspp.shape
        dec0_fine = self.bot_fine(m2)
        dec0_fine=F.interpolate(dec0_fine,size=torch.Size([w, h]), mode='bilinear', align_corners=True)
        # seg_edge=torch.cat([x_edge,seg_edge],dim=1)

        seg_edge = self.edge_fusion(torch.cat([seg_edge, dec0_fine], dim=1))
        seg_edge=self.se2(seg_edge)
        seg_edge=self.upsample4(seg_edge)
        # seg_body=self.upsample4(seg_body)
        # aspp=torch.cat([aspp,seg_body],dim=1)
        # aspp=self.upsample5(aspp)
        # seg_out = seg_edge + Upsample(seg_body, [size, size])
        # seg_out = torch.cat([aspp, seg_body],dim=1)
        seg_final=aspp+seg_edge
        # seg_final = self.final_seg(seg_out)
        
        
        # seg_edge_out = Upsample(seg_edge_out, [size, size])
        # seg_edge_out = self.sigmoid_edge(seg_edge_out)
        output = Upsample(seg_final, [256, 256])
        # for i in range(x.shape[0]):
        #     img3=VisualizeFeatureMapPCA(seg_edge[i])
        #     cv2.imwrite('result/'+str(name[i])+'_output.png',img3)
        # output=F.interpolate(output,size=torch.Size([256,256]),mode='bilinear',align_corners=True)
        
        output=self.out(output)
        # output=self.se(output)
        # output+=residual
        # print(output.shape)
        # output=self.relu(output)
        # print(output.shape)
        self.number+=1

        return output

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x=self.relu(x)

        # relu自己修改
        return x

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)



class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2,bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2,bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, padding=1,bias=False),
            norm_layer(inplane),
            # nn.ReLU(inplace=True),
        )
        self.xflow = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.flow_make = nn.Conv2d(inplane  , 2, kernel_size=3, padding=1, bias=False)
        self.se=SELayer(inplane,reduction=16)
        self.se1=SELayer(inplane*2,reduction=16)
        self.upSample=BasicConv2d(32, 32, 3, padding=1)
        self.upsample2=BasicConv2d(128,64,3,padding=1)

    def forward(self, x,x2=None):
        size = x.size()[2:]
        # seg_down = self.down(x)   #512,53,53
        # seg_down = F.interpolate(seg_down, size=size, mode="bilinear", align_corners=True)
        # flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        # flow = self.flow_make(x)
        x_avg=torch.mean(x,dim=1,keepdim=True)
        x_max,_=torch.max(x,dim=1,keepdim=True)
        flow=torch.cat([x_max,x_avg],dim=1)

        if x2!=None:
            x2=self.upSample(x2)
            seg_flow_warp = self.flow_warp(x, flow, size,x2)
            # x2=self.xflow(x2)
            # x=x+x2
        else:
            seg_flow_warp = self.flow_warp(x, flow, size)

        

        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge
        
    def flow_warp(self, input, flow, size,x=None):
        
        out_h, out_w = size
        n, c, h, w = input.size()
        input=self.se(input)
        
        # x=F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # torch.linespace(start,end,steps,dtype)将start到end等分成steps份，类型为dtype
        # torch.view将原本的数据整理成新的形状，-1为不确定行数，但是确定列数
        # torch.repeat(a,b)把行复制a次，列复制b次
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        if x!=None:
            x2=self.xflow(x)
            input=input+x2

        output = F.grid_sample(input, grid,mode='bilinear',align_corners=True)
        return output
        

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': True}
    Net = PAFNet()
    Net = Net.double()

    x  = np.random.rand(2,3,256,256)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)

 