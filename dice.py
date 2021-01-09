import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
import torch.nn.functional as F
import cv2
import os
import numpy as np

for i in range(0, 129):
    s2 = cv2.imread('save3/iter%d-mask.jpg' % i, 0)  # 模板
    row, col = s2.shape[0], s2.shape[1]
    s1 = cv2.imread('save3/iter%d-target.jpg' % i, 0)  # 读取配准后图像
    d = []
    s = []
    for r in range(row):
        for c in range(col):
            if s1[r][c] == s2[r][c]:  # 计算图像像素交集
                s.append(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
    d.append(2*m1/m2)
    msg = "这是第{}张图的dice系数".format(i) + str(2 * m1 / m2)
    # print(2*m1/m2)
    print(msg)
    with open('dice.txt', 'a') as file_object:
        file_object.write(str(i))
        file_object.write(" ")
        file_object.write(str(2 * m1 / m2))
        file_object.write("\n")
# print(d)
