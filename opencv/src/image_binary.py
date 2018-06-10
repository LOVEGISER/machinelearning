# -*- coding: utf-8 -*-
"""
Created on 2018-06-10 13:13:00
@author: Alex
"""
#encoding:utf-8
import numpy as np
import cv2

image = cv2.imread("resources/plate.png")
#二值化
ret,thresh1=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3=cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
ret,thresh4=cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]


# 中值滤波
img = cv2.medianBlur(image,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#11 为 Block size, 2 为 C 值
mg_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
th2 = cv2.adaptiveThreshold(mg_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(mg_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

while True:
    cv2.imshow('th1', mg_grey)
    cv2.imshow('th2', th2)
    cv2.imshow('th3', th3)
    cv2.imshow('thresh1',thresh1)
    cv2.imshow('thresh2', thresh2)
    cv2.imshow('thresh3', thresh3)
    cv2.imshow('thresh4', thresh4)
    cv2.imshow("thresh5",thresh5)
    if cv2.waitKey(10) & 0xFF == ord(' '):
        break

