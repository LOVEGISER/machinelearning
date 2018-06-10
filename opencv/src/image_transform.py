# -*- coding: utf-8 -*-
"""
Created on 2018-06-10 13:13:00
@author: Alex
"""
#encoding:utf-8
import numpy as np
import cv2
image = cv2.imread("resources/anna.jpeg")
#平移
matrix = np.float32([[1,0,25],[0,1,50]])#平移矩阵1：向x正方向平移25，向y正方向平移50
moved = cv2.warpAffine(image,matrix,(image.shape[1],image.shape[0]))

#旋转
rows,cols,ch=image.shape
# 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
transform_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
# 第三个参数是输出图像的尺寸中心
transform=cv2.warpAffine(image,transform_matrix,(2*cols,2*rows))

#仿射变换-affine
pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[10,100],[200,50],[100,250]])
affine=cv2.getAffineTransform(pts1,pts2)
print(affine)
affined=cv2.warpAffine(image,affine,(cols,rows))
#透视变换
perspective_pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
perspective_pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
perspective_M=cv2.getPerspectiveTransform(perspective_pts1,perspective_pts2)
perspective=cv2.warpPerspective(image,perspective_M,(300,300))
while True:
    cv2.imshow('original',image)
    cv2.imshow('moved', moved)
    cv2.imshow('transform', transform)
    cv2.imshow('affined', affined)
    cv2.imshow("perspective",perspective)
    if cv2.waitKey(10) & 0xFF == ord(' '):
        break
