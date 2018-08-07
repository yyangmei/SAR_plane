import os
import cv2
import cPickle
import numpy as np
import sys
from pascal_voc import remote_data
from th_class import *
import pdb

# ------------
# ------------



'''
def test_net(annopath,imagesetfile):
    """Test a Fast R-CNN network on an image database."""
    output_dir = '/share/home/math4/oldcaffe/wangjiaojuan/Mymap/results/newremotedata/'
    print 'Evaluating detections'
    imdb.evaluate_detections(output_dir,annopath,imagesetfile)


if __name__ == '__main__':
    # load data
    imdb = remote_data(pascal_path='/share/home/math4')
    annopath = "/share/home/math4/label"
    imagesetfile = "/share/home/math4/img"
    imdb.competition_mode(on=True)
    # evaluation
    test_net(annopath,imagesetfile)


'''
def test_net(annopath,imagesetfile):
    """Test a Fast R-CNN network on an image database."""
    output_dir = '/share/home/math8/ym/R-FCN/eval/results/newremotedata/'
    print 'Evaluating detections'
    imdb.evaluate_detections(output_dir,annopath,imagesetfile)


if __name__ == '__main__':
    # load data
    imdb = remote_data(pascal_path='/share/home/math4')
    annopath = "/share/home/math4/label"
    imagesetfile = "/share/home/math4/img"
    imdb.competition_mode(on=True)
    # evaluation
    savedir='/share/home/math8/ym/R-FCN/eval/results/newremotedata/'
    path='/share/home/math8/ym/R-FCN/eval/results/'
    for th1 in xrange(0,10,1):
        th=float(th1)/10
        #pdb.set_trace()
        print ".....................ship.........conf thesh is ............",th
        calss(th,path,savedir)
        test_net(annopath,imagesetfile)
