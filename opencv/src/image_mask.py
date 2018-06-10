# -*- coding: utf-8 -*-
"""
Created on 2018-06-10 13:13:00
@author: Alex
"""
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while(1):
    # 获取每一帧
    ret,frame=cap.read()
    # 转换到 HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # 设定蓝色的阈值
    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])
    # 跟踪对象的 HSV 值
    # green = np.uint8([[[0, 255, 0]]])
    # hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    # 现在你可以分别用[H - 100， 100， 100] 和[H + 100， 255， 255] 做上下阈值。
    # 根据阈值构建掩模
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)
    # 显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(5)&0xFF
    if k==27:
     break
# 关闭窗口
cv2.destroyAllWindows()