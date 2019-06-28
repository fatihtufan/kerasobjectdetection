#!/usr/bin/python
#-*- coding: utf-8 -*-
# -*- coding: cp1254 -*-
# coding: latin1
"""
Created on Tue Jun 25 15:05:28 2019

@author: Ogün Can KAYA
"""

# Yeni bir kayıt eklemek için yapılacak işlemler.
import cv2
import os
import numpy as np

cap = cv2.VideoCapture(0)

name = input(u"İsminizi girin: ")
os.mkdir('img/'+name)

#uniName=unicode(name,"utf-8")

width_height = 300

def resize(max_height: int, max_width: int, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]

    if max_height < height or max_width < width:
        if width < height:
            scaling_factor = max_width / float(width)
        else:
            scaling_factor = max_height / float(height)

        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

def crop_center(frame: np.ndarray) -> np.ndarray:
    short_edge = min(frame.shape[:2])
    yy = int((frame.shape[0] - short_edge) / 2)
    xx = int((frame.shape[1] - short_edge) / 2)
    crop_img = frame[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

    start = int((frame.shape[1] - frame.shape[0]) / 2)
    end = int(frame.shape[1] - (frame.shape[1] - frame.shape[0]) / 2)
    return frame[:, start:end]






i=0
while(True):
    i = i + 1
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    frame = resize(width_height, width_height, frame)
    frame = crop_center(frame)
     
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    cv2.imshow('frame', rgb)
   
    cv2.imwrite('img/' + (name) + '/' + str(i) + '.jpg', frame)
    cv2.waitKey(1) & 0xFF
    if i>=200:
        break
        
    
       # if cv2.waitKey(1) & 0xFF == ord('q'):
       # out = cv2.imwrite("img/"+(name)+".jpg", frame)
        #break
cap.release()
cv2.destroyAllWindows()