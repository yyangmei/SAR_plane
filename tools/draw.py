# -*- coding: utf-8 -*-
#这个代码是将标注的框画在图上

# In[1]:
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os


##
saveimg="/share/home/math8/ym/R-FCN/result_img/"
imgdir="/share/home/math4/img/"
############################
label1="/share/home/math8/ym/R-FCN/eval/results/det_trainval_ship_36000.txt"  
file=open(label1,'r')
line=file.readlines()
i=0
print 'objects number:',len(line)
while i<len(line):
    print 'object',i,'/',len(line)
    if len(line[i].split(' '))>=6:
        fi=line[i].split(' ')[0]
        if os.path.exists(saveimg+fi):
           im1=cv2.imread(saveimg+fi)
        else:
           im1=cv2.imread(imgdir+fi)
        score=float(line[i].split(' ')[1])
        label=1
        if score>0.5:
           xmin=int(float(line[i].split(' ')[2]))
           ymin=int(float(line[i].split(' ')[3]))
           xmax=int(float(line[i].split(' ')[4]))
           ymax=int(float(line[i].split(' ')[5]))
           cv2.rectangle(im1,(xmin,ymin),(xmax,ymax), (0, 255, 0), 1)
           cv2.putText(im1, '%s' % (score), (xmin, ymin), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
           cv2.imwrite("/share/home/math8/ym/R-FCN/result_img/"+fi,im1)
        i=i+1
    else:
        i=i+1
print 'Done all images.........................................'
############################
# label1="/share/home/math4/oldcaffe/wangjiaojuan/plane_boat_test/result_label/det_trainval_2.txt"  
# file=open(label1,'r')
# line=file.readlines()
# i=0
# while i<len(line):
    # if len(line[i].split(' '))>=6:
        # fi=line[i].split(' ')[0]
        # if os.path.exists(saveimg+fi):
           # im1=cv2.imread(saveimg+fi)
        # else:
           # im1=cv2.imread(imgdir+fi)
        # label=int(float(line[i].split(' ')[1]))
        # xmin=int(float(line[i].split(' ')[2]))
        # ymin=int(float(line[i].split(' ')[3]))
        # xmax=int(float(line[i].split(' ')[4]))
        # ymax=int(float(line[i].split(' ')[5]))
        # cv2.rectangle(im1,(xmin,ymin),(xmax,ymax), (0, 255, 0), 1)
        # cv2.putText(im1, '%s' % (label), (xmin, ymin), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
        # cv2.imwrite("/share/home/math4/oldcaffe/wangjiaojuan/plane_boat_test/allimg/"+fi,im1)
        # i=i+1
    # else:
        # i=i+1