# -*- coding: utf-8 -*-
# from __future__ import print_function, division
from net.net2d.unet2d import UNet2D
from net.net2d.unet2d import UNet2D
from net.net2d.paf_net import PAFNet
from net.net2d.cople_net import COPLENet
from net.net2d.unet2d_attention import AttentionUNet2D
from net.net2d.unet2d_nest import NestedUNet2D
from net.net2d.unet2d_scse import UNet2D_ScSE
from net.net3d.unet2d5 import UNet2D5
from net.net3d.unet3d import UNet3D
from net.net3d.unet3d_scse import UNet3D_ScSE

SegNetDict = {
	'UNet2D': UNet2D,
	'COPLENet': COPLENet,
	'AttentionUNet2D': AttentionUNet2D,
	'NestedUNet2D': NestedUNet2D,
	'UNet2D_ScSE': UNet2D_ScSE,
	'UNet2D5': UNet2D5,
	'UNet3D': UNet3D,
	'UNet3D_ScSE': UNet3D_ScSE,
	'PAFNet': PAFNet
	}
