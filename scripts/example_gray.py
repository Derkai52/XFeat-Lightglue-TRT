import os
import numpy as np
import torch
import tqdm
import cv2

from modules.xfeat import XFeat

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeat()

# read image
img = cv2.imread('/home/tk/Desktop/image.png', cv2.IMREAD_GRAYSCALE)

# 使用cv2.resize()进行图像缩放
dim = (640, 480)
img = cv2.resize(img, dim)

# convert to tensor [1, 1, H, W]
x = torch.tensor(img, dtype = torch.float32)
x = x[None, None, :, :] / 255.0

print(x.shape)

points = xfeat.detectAndCompute(x, top_k = 4096)[0]

print("----------------")

# draw on image
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for p in points['keypoints']:
    cv2.circle(cimg, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

cv2.imshow('image', cimg)
cv2.waitKey(0)

# print keypoints
for p in points['keypoints']:
    print("{}, {}".format(p[0], p[1]))

# print descriptors
for d in points['descriptors']:
    print(d)