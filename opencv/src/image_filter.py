# -*- coding: utf-8 -*-
"""
Created on 2018-06-10 13:13:00
@author: Alex
"""
#encoding:utf-8
import numpy as np
import cv2

image = cv2.imread("resources/plate.png")
#图像平滑
#定义5*5 的滤波器核
#2d卷积
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(image,-1,kernel)
#均值滤波
blur = cv2.blur(image,(5,5))
#高斯（正太）分布滤波
#0 是指根据窗口大小（5,5）来计算高斯函数标准差
gaussianBlur = cv2.GaussianBlur(image,(5,5),0)
median = cv2.medianBlur(image,5)
while True:
    cv2.imshow('image', image)
    cv2.imshow('dst', dst)
    cv2.imshow('blur', blur)
    cv2.imshow("gaussianBlur",gaussianBlur)
    cv2.imshow('median', median)
    if cv2.waitKey(10) & 0xFF == ord(' '):
        break

