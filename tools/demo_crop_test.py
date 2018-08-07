#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""



import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms,soft_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import ipdb
import argparse

import codecs
import chardet



CLASSES = ('__background__',
           'airplane')
#当类别改变时需要改一下，在给txt文件起名字以及画图的时候用到

CONF_THRESH = [0]
#CONF_THRESH有两个地方会用到，一个当网络刚输出来时，针对每个类别，仅仅保留那些置信概率大于CONF_THRESH的框

NMS_THRESH = [0.3]
#NMS_THRESH是指对不同的类它们的nms阈值

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

# cfg.TEST.RPN_POST_NMS_TOP_N= 300   #RPN网络内部参数，RPN完事以后要留下的proposal的数量
# cfg.TEST.SCALES = (1000,)          
# cfg.TEST.MAX_SIZE = 1000
del_theta=0.55   #后处理参数，指两个框的交比上各自的面积大于del_theta时，会舍掉这个框
                 #（存在一个问题是：当两个框大小差不多然后相交部分比较大时有可能都被舍弃）不存在！每次只删除一个但是对于这种情况可能把概率大的给删了
                 #所以还是应该先做nms
del_theta_p=0.8
#del_theta2=[1.8,1.5,15,10,10,5,2,3]
del_theta2=[10]

prototxt = "models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt"

caffemodel = "output/rfcn_end2end_ohem/voc_2007_trainval/resnet101_deform_fintune36000_iter_22000.caffemodel"

#测试图像地址
im_path="/share/home/math8/ym/R-FCN/data/demo/"

#测试结果txt保存地址
result_path="eval/results/"

#测试结果图像保存地址
save_path="20000test/"  

##裁减图像的的函数
def crop_im(crop_size,crop_overlap,im):
    w = im.shape[0]
    h = im.shape[1]
    w_num=(w-crop_size)/(crop_size-crop_overlap)+1
    h_num=(h-crop_size)/(crop_size-crop_overlap)+1       
    index = []
    for j in range(1,w_num+1):
     for k in range(1,h_num+1):
         start_x=(crop_size-crop_overlap)*(j-1)
         start_y=(crop_size-crop_overlap)*(k-1)
         end_x=start_x+crop_size
         end_y=start_y+crop_size
         box=(start_x,start_y,end_x,end_y)
         box=(box,h_num*(j-1)+k)#((start_x,start_y,end_x,end_y),h_num*(j-1)+k)
         index.append(box)
         
    h_num_1=0
    if end_x < w:
        h_num_1=(h-crop_size)/(crop_size-crop_overlap)+1
        start_x_1=w-crop_size
        end_x_1=w
        for l in range(1,h_num_1+1):
             start_y_1=(crop_size-crop_overlap)*(l-1)
             end_y_1=start_y_1+crop_size
             box=(start_x_1,start_y_1,end_x_1,end_y_1)
             box=(box,w_num*h_num+l)
             index.append(box)
             
     
    w_num_1=0
    if end_y < h:
        w_num_1=(w-crop_size)/(crop_size-crop_overlap)+1
        start_y_2=h-crop_size
        end_y_2=h
        for m in range(1,w_num_1+1):
             start_x_2=(crop_size-crop_overlap)*(m-1)
             end_x_2=start_x_2+crop_size
             box=(start_x_2,start_y_2,end_x_2,end_y_2)
             box=(box,w_num*h_num+h_num_1+m)
             index.append(box)
    
    
    if  end_x < w and end_y < h:
        start_x_3=w-crop_size
        start_y_3=h-crop_size
        end_x_3=w
        end_y_3=h
        box=(start_x_3,start_y_3,end_x_3,end_y_3)
        box=(box,w_num*h_num+h_num_1+w_num_1+1)
        index.append(box)
    return index


##后处理的函数，（1）两个框的交比上得分低的循环框的面积，判断是否大于阈值，大于阈值则去掉得分低的循环框
              ##（2）两个框的交比上得分高的固定框的面积，判断是否大于阈值，大于阈值则去掉得分低的循环框
def postprocess(dets,del_theta,del_theta_p):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:,4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep2 = []
    while order.size > 0:
        i = order[0]
        keep2.append(i)
        areas_i=(x2[i]-x1[i])*(y2[i]-y1[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter/areas[order[1:]]
        ovr2=inter/areas_i
        #from ipdb import set_trace
        #set_trace()
        
        
        
        inds2 = np.where(ovr2 <= del_theta_p)[0]
        inds = np.where(ovr <= del_theta)[0]
        ind3=np.array([l for l in inds if l in inds2])
        if len(ind3)==0:
           break
        

        # from ipdb import set_trace
        # set_trace()
        order = order[ind3 + 1]

    return keep2

##后处理的函数，计算框的长宽比去掉长宽比不恰当的框
def postprocess2(dets,del_theta2):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    #scores = dets[:,4]
    z=(x2-x1)/(y2-y1)
    #from ipdb import set_trace
    #set_trace() 
    inds = np.where((z < del_theta2) & (z>(1/del_theta2))  )[0]
    return inds
    
    
    
    

##加载图像，调用裁减函数，调用caffe进行测试，把测试结果合到原图上，调用后处理函数以及调用nms或者softnms，最后保存
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(im_path, image_name)
    timer = Timer()
    timer.tic()
    im = cv2.imread(im_file)
    timer.toc()
    print ('reading image took {:.3f}s for detection').format(timer.total_time)
    crop_size=6000 #裁减图像大小
    crop_overlap=100 #裁减图像的重叠区域
    # ipdb.set_trace()
    if im.shape[0]>crop_size and im.shape[1]>crop_size:
        index=crop_im(crop_size,crop_overlap,im)
        all_dets=[[]for _ in xrange(2)]       
        #print index
        for im_index in range(0,len(index)):   
            start_x=index[im_index][0][0]
            start_y=index[im_index][0][1]
            end_x=index[im_index][0][2]
            end_y=index[im_index][0][3]     
            scores, boxes = im_detect(net, im[start_x:end_x,start_y:end_y])
            
            # skip j = 0, because it's the background class
            for class_index in xrange(1, 2):
                inds = np.where(scores[:, class_index] > CONF_THRESH[class_index-1])[0]  #confidence thresh
                if len(inds)==0:
                   continue
                # from ipdb import set_trace
                # set_trace() 
                cls_scores = scores[inds, class_index]
                #cls_boxes = boxes[inds, class_index * 4:(class_index + 1) * 4]
                cls_boxes = boxes[inds, 4:8]
                #from ipdb import set_trace
                #set_trace() 
                ###函数im_detect的输出是什么样的？这里为啥要乘上4？？？？？？？？？？？
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                #后处理函数
                #cls_dets=postprocess(cls_dets,del_theta)    
                #softnms，如果不使用该方法可以注释掉，这个是faster自带的softnms，但是
                #它是将所有类不加区分放在一起进行softnms，而且所有的类共用一个置信概率 
                #keep = soft_nms(cls_dets, sigma=0.5, Nt=0.3, threshold=0.001, method=2)
                #2是高斯，1是线性，设其他是nms
                #nms，如果不使用该方法也注释掉，它和soft_nms二选一
                #from ipdb import set_trace
                #set_trace() 
                #keep = nms(cls_dets, NMS_THRESH[class_index-1]) #nms thresh
                #cls_dets = cls_dets[keep, :]
                ##index的每一行的结构((start_x,start_y,end_x,end_y),h_num*(j-1)+k)
                cls_dets[:,:1]=(cls_dets[:,:1]+index[im_index][0][1])
                cls_dets[:,1:2]=(cls_dets[:,1:2]+index[im_index][0][0])
                cls_dets[:,2:3]=(cls_dets[:,2:3]+index[im_index][0][1])
                cls_dets[:,3:4]=(cls_dets[:,3:4]+index[im_index][0][0])
                all_dets[class_index].append(cls_dets.tolist())
 
        # from ipdb import set_trace
        # set_trace() 
        for j in xrange(1, 2):
            if len(all_dets[j])==0:
               continue
            whole_dets=np.vstack(all_dets[j])
            
            
            ##后处理1
            # keep2=postprocess(whole_dets,del_theta,del_theta_p)#1111111111111
            
          
            #keep = soft_nms(whole_dets, sigma=0.5, Nt=0.3, method=2, threshold=0.001) 
            ##后处理2，一般NMS，上面用的是soft-NMS
            whole_dets=whole_dets.astype(np.float32, copy=False)
            keep = nms(whole_dets, NMS_THRESH[class_index-1]) #111111111111
            #whole_dets=all_dets_pos[keep]#11111111111111111
            ##后处理3
            # whole_dets1=all_dets_pos[keep]
            # ind=postprocess2(whole_dets1,del_theta2[j-1])
            whole_dets=whole_dets[keep] 
            
            ##把最终结果按得分排序，不需要所以注释掉
            # a_arg=np.argsort(-whole_dets[:,4])
            # whole_dets=whole_dets[a_arg]  #rank

            if os.path.exists(result_path):
                pass
            else:
                os.mkdir(result_path)
            file1=open(result_path+'det_test_'+CLASSES[j]+'.txt','a')
            for i in range(whole_dets.shape[0]):
                bbox = tuple(int(np.round(x)) for x in whole_dets[i, :4])
                score = whole_dets[i, -1]
                
                ##画图
                if score>0.5:
                    cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                    cv2.putText(im, '%s: %.3f' % (CLASSES[j], score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                         1.0, (0, 0, 255), thickness=1)

                # if image_name.find('.tiff') == -1:  # this img is png or tif
                    # im_name=image_name[:-4]
                # else:  #this img is tiff
                    # im_name=image_name[:-5]         
                line=image_name+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n'
                file1.write(line)
				#file1.write(line)
            file1.close()
    else:
        scores, boxes = im_detect(net, im)
        # from ipdb import set_trace
        # set_trace()         
        for class_index in xrange(1, 2):
            #print(class_index)
            inds = np.where(scores[:, class_index] > CONF_THRESH[class_index-1])[0]  #confidence thresh
            if len(inds)==0:
                continue
            #############################
            #print(inds)
            ###############################
            cls_scores = scores[inds, class_index]
            cls_boxes = boxes[inds, 4:8]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            # keep2=postprocess(cls_dets,del_theta,del_theta_p)
            # all_dets_pos=cls_dets[keep2]
            #keep = soft_nms(cls_dets, sigma=0.5, Nt=0.3, method=2, threshold=0.001) 
            keep = nms(cls_dets, NMS_THRESH[class_index-1]) #nms thresh
            cls_dets = cls_dets[keep]
            
            # ind=postprocess2(cls_dets,del_theta2[class_index-1])
            # cls_dets=cls_dets[ind]
            # a_arg=np.argsort(-cls_dets[:,4])
            # cls_dets=cls_dets[a_arg]

            if os.path.exists(result_path):
                pass
            else:
                os.mkdir(result_path)
            
            file1=open(result_path+'det_test_'+CLASSES[class_index]+'.txt','a')
            for i in range(cls_dets.shape[0]):
                bbox = tuple(int(np.round(x)) for x in cls_dets[i, :4])
                score = cls_dets[i, -1]
                if score>0.5:
                    cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                    cv2.putText(im, '%s: %.3f' % (CLASSES[class_index], score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                         1.0, (0, 0, 255), thickness=1)
                # if image_name.find('.tiff') == -1:  # this img is png or tif
                    # im_name=image_name[:-4]
                # else:  #this img is tiff
                    # im_name=image_name[:-5]         
                
                line=im_name+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n'
                file1.write(line)
            file1.close()
            
    
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)        
    cv2.imwrite(os.path.join(save_path+'/'+image_name),im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=2, type=int)#默认使用gpu 0
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

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
    
    time = 0
    im_names = os.listdir(im_path)
    num = len(im_names)
    # num=len(filenames)
    for i,im_name in enumerate(im_names):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'This is the image', i, '/',num,':', im_name
        timer = Timer()
        timer.tic()
        demo(net, im_name)
        timer.toc()
        print ('Detection took {:.3f}s for detection').format(timer.total_time)
        global time
        time += timer.total_time
    print 'Average time = ',time/num



