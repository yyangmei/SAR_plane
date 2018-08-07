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
def calss(th,path,savedir):
    # if os.path.exists(savedir+'det_trainval_aeroplane.txt'):
       # os.remove(savedir+'det_trainval_aeroplane.txt')
       # os.remove(savedir+'aeroplane_pr.pkl')
    if os.path.exists(savedir+'det_trainval_ship.txt'):
       os.remove(savedir+'det_trainval_ship.txt')  
       os.remove(savedir+'ship_pr.pkl')  
    #####del plane
    # planelabel=path+'det_trainval_ship_80000.txt'
    # file1=open(savedir+'det_trainval_aeroplane.txt','a')
    # file=open(planelabel,'r')
    # line=file.readlines()
    # i=0
    # while i<len(line):
        # if len(line[i].split(' '))>=6:
            # name=line[i].split(' ')[0]
            # score=float(line[i].split(' ')[1])
            # xmin=line[i].split(' ')[2]
            # ymin=line[i].split(' ')[3]
            # xmax=line[i].split(' ')[4]
            # ymax=line[i].split(' ')[5]
            # newline1=name+' '+str(score)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)
            # if score>th:
               # file1.write(newline1)
            # i=i+1
        # else:
            # i=i+1
    # file1.close()
    #############################
    #####del plane
    boatlabel=path+'det_trainval_ship_38000.txt'
    file2=open(savedir+'det_trainval_ship.txt','a')
    file=open(boatlabel,'r')
    line=file.readlines()
    i=0
    while i<len(line):
        if len(line[i].split(' '))>=6:
            name=line[i].split(' ')[0]
            score=float(line[i].split(' ')[1])
            xmin=line[i].split(' ')[2]
            ymin=line[i].split(' ')[3]
            xmax=line[i].split(' ')[4]
            ymax=line[i].split(' ')[5]
            newline1=name+' '+str(score)+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)
            if score>th:
               file2.write(newline1)
            i=i+1
        else:
            i=i+1
    file2.close()