from code import interact
import os
from unittest import result
import cv2
import numpy as np
from tqdm import  tqdm

def class_wise(predict,gth):
    a,b,c=predict.shape
    predict=predict.transpose(2,0,1)
    gth=gth.transpose(2,0,1)
    predict=predict[0]
    gth=gth[0]
    for i in range(a):
        for j in range(b):
                if predict[i,j]>=127:
                    predict[i,j]=1
                else:
                     predict[i,j]=0
    for i in range(a):
        for j in range(b):
                if gth[i,j]>=127:
                    gth[i,j]=1
                else:
                    gth[i,j]=0
    zong=np.ones((256,256))
    yuce=np.sum(predict)
    zhenshi=np.sum(gth)
    interact=np.sum(predict*gth)
    tp=interact
    fn=yuce-interact
    fp=zhenshi-interact
    zong=np.sum(zong)
    tn=zong-yuce-zhenshi+interact
    return tp,fn,fp,tn


dice=[]
acc=[]
sen=[]
spec=[]
prec=[]
num=[]
result_path="result/"
# gth_path="dataset/picture/data/label/"
gth_path="dataset/picture/data/label/"
list=os.listdir(result_path)
for path in tqdm(list):
    # path='1662.png'
    img=cv2.imread(result_path+path)
    gth=cv2.imread(gth_path+path)
    tp,fn,fp,tn=class_wise(img,gth)
    dice.append(2*tp/(2*tp+fn+fp))
    prec.append(tp/(tp+fp))   
    sen.append(tp/(tp+fn))
    acc.append((tp+tn)/(tp+fp+fn+tn))
    spec.append(tn/(tn+fp))
    
    # print(dice)
# for i in range(len(dice)):
#     if dice[i]!=0:
#         num.append(dice[i])
# print(np.mean(num))
print(np.mean(dice))
print(np.mean(acc))
print(np.mean(sen))
print(np.mean(spec))
print(np.mean(prec))

                