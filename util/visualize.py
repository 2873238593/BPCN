from sklearn .decomposition import PCA
import visdom
import torch
import torch.nn.functional as F
import cv2
import numpy as np

'''Visualize Flow Start'''

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


'''Visualize Feature Map PCA Start'''


def VisualizeFeatureMapPCA(input_tensor, title='default', num=1, scale=1):

    feature = input_tensor.data.cpu().numpy()
    # img_out = np.mean(feature, axis=0)
    feature = feature
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    pca = PCA(n_components=3)
    pca.fit(img_out)
    img_out_pca = pca.transform(img_out)
    img_out_pca = img_out_pca.transpose(
        1, 0).reshape(3, h, w).transpose(1, 2, 0)

    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(
        img_out_pca, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    img_out = np.array(img_out_pca, dtype=np.uint8)
    img_out=cv2.cvtColor(img_out,cv2.COLOR_RGB2BGR)
    # img_out = img_out.transpose(2, 0, 1)

    # img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    # vis3.image(img_out, win=title, opts=dict(
    #     title=title))
    # img_out = img_out.swapaxes(0, 1)
    # img_out = img_out.swapaxes(1, 2)
    return img_out


'''Visualize Feature Map PCA End'''

'''Visualize Feature Map PCA DIfferent Color Start'''


def VisualizeFeatureMapPCA_Alter(input_tensor, title='default', num=0, scale=1):

    feature = input_tensor.data.cpu().numpy()
    # img_out = np.mean(feature, axis=0)
    feature = feature[num]
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    pca = PCA(n_components=3)
    pca.fit(img_out)
    img_out_pca = pca.transform(img_out)
    img_out_pca = img_out_pca.transpose(
        1, 0).reshape(3, h, w).transpose(1, 2, 0)
    img_out_pca = 255-img_out_pca
    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(
        img_out_pca, (w, h), interpolation=cv2.INTER_LINEAR)
    img_out = np.array(img_out_pca, dtype=np.uint8)
    img_out = img_out.transpose(2, 0, 1)

    # img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    vis3.image(img_out, win=title, opts=dict(
        title=title))
    return img_out


'''Visualize Feature Map PCA End'''

vis3 = visdom.Visdom(port=8098)


def VisualizeChannel(input_tensor, title='default', num=0, scale=1):
    out = input_tensor
    h, w = out.size()[-2:]
    out = F.interpolate(out, size=(h*scale, w*scale),
                        mode='bilinear', align_corners=True)
    vis3.images(torch.unsqueeze(out[num], dim=1), win=title, opts=dict(
        title=title))


def VisualizeSingleChannel(input_tensor, title="default", num=0, scale=1):
    out = input_tensor  # shape: [b,c,h,w]
    h, w = out.size()[-2:]
    out = F.interpolate(out, size=(h*scale, w*scale),
                        mode='bilinear', align_corners=True)
    out = out[num]  # shape : [c,h,w]
    out = torch.mean(out, dim=0)  # shape:[h,w]
    out = torch.unsqueeze(out, dim=0)
    # for i in range(input_tensor.size()[0]):
    vis3.image(out, win=title, opts=dict(
        title=title))
