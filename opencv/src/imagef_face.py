# -*- coding: utf-8 -*-
"""
Created on 2018-06-10 13:13:00
@author: Alex
"""
import cv2
import numpy as np

cv2.namedWindow("test")
cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("resources/haarcascades/haarcascade_frontalface_default.xml")

while True:
   success, frame = cap.read()
   faces = classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

   for (x, y, w, h) in faces:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

   cv2.imshow("test", frame)
   key = cv2.waitKey(10)
   c = chr(key & 255)
   if c in ['q', 'Q', chr(27)]:
    break
cv2.destroyWindow("test")