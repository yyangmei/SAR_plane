#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import soft_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
#os.environ['CUDA_VISIBLE_DEVICES']='1'

CLASSES = ('__background__',
           'ship')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_deform_soft_iter_32000.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}
save_label_dir = 'eval/results/'
save_img_dir = '58000test/'
file1=open(save_label_dir+'det_trainval_airplane.txt','a')
file2=open(save_label_dir+'det_trainval_ship_58000.txt','a')



def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    if os.path.exists(save_img_dir+image_name):
        im = cv2.imread(save_img_dir+image_name)
    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    # if class_name == '1':
        # for i in inds:
            # bbox = dets[i, :4]
            # score = dets[i, -1]
            #print '111111111111111111111111',image_name,class_name,score,bbox
            # newline1=str(image_name)+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n' 
            # file1.write(newline1)
            # cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0, 255, 0), 1)                                                        ##########...............................
            # cv2.putText(im, '%s: %f' % (str(class_name),score), (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
    if class_name == 'ship':
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            #print '2222222222222222222222222222',image_name,class_name,score,bbox
            newline1=str(image_name)+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n' 
            file2.write(newline1)
            # if score > 0.5:
                # cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (0, 255, 0), 1)                                                        ##########...............................
                # cv2.putText(im, '%s: %f' % (str(class_name),score), (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness=1)
    # cv2.imwrite(save_img_dir+image_name,im)
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('/share/home/math4/img/', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    global time
    time += timer.total_time
    CONF_THRESH =0 #-np.inf
    #NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = soft_nms(dets, sigma=0.5, Nt=0.3, method=1, threshold=0.5)  
        dets = dets[keep, :]
        vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)
    #cv2.imwrite(save_img_dir+fi+'.jpg',im)
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=3, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            # 'rfcn_end2end', 'test_agnostic.prototxt')
    # caffemodel = os.path.join('output/rfcn_end2end_ohem/voc_2007_trainval',
                              # NETS[args.demo_net][1])
    prototxt = 'models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt'
    caffemodel = 'output/rfcn_end2end_ohem/voc_2007_trainval/resnet101_deform_fintune36000_iter_22000.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
        
    rootdir = '/share/home/math4/img/'
    #rootdir = '/share/home/math8/ym/R-FCN/testimg/'
    im_names = os.listdir(rootdir)
    print 'The number of test images is:', len(im_names)
    time = 0
    for i,im_name in enumerate(im_names):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'This is the image', i, ':', im_name
        demo(net, im_name)

    print 'Average time = ',time/len(im_names)

    plt.show()
