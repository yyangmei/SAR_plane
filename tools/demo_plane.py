#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
           'plane','ship','airport')
#当类别改变时需要改一下，在给txt文件起名字以及画图的时候用到   

CONF_THRESH = [0,0,0]
#CONF_THRESH有两个地方会用到，一个当网络刚输出来时，针对每个类别，仅仅保留那些置信概率大于CONF_THRESH的框

NMS_THRESH = [0.3,0.3,0.3]
#NMS_THRESH是指对不同的类它们的nms阈值

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

del_theta=0.55   #后处理参数，指两个框的交比上各自的面积大于del_theta时，会舍掉这个框
                 #（存在一个问题是：当两个框大小差不多然后相交部分比较大时有可能都被舍弃）不存在！每次只删除一个但是对于这种情况可能把概率大的给删了
                 #所以还是应该先做nms
del_theta_p=0.8
#del_theta2=[1.8,1.5,15,10,10,5,2,3]
del_theta2=[10]

root = 'YM_SAR/'

prototxt_airport = root+'models/pascal_voc/ResNet-101/ymsar_end2end/test_airport.prototxt'
caffemodel_airport = root+'output/ymsar_end2end_ohem/voc_2007_trainval/Airport.caffemodel'
    
prototxt_plane = root+'models/pascal_voc/ResNet-101/ymsar_end2end/test_agnostic.prototxt'
caffemodel_plane = root+'output/ymsar_end2end_ohem/voc_2007_trainval/Plane.caffemodel'

#测试图像地址
im_path=root + "data/demo/"

#测试结果txt保存地址
result_path = root+"results/"

#测试结果图像保存地址
save_path = root+"resultsimg/" 

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

def demo_plane(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    crop_size=2000 #裁减图像大小
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
                inds = np.where(scores[:, 1] > CONF_THRESH[0])[0]  #confidence thresh
                if len(inds)==0:
                   continue
                # from ipdb import set_trace
                # set_trace() 
                cls_scores = scores[inds, 1]
                #cls_boxes = boxes[inds, class_index * 4:(class_index + 1) * 4]
                cls_boxes = boxes[inds, 4:8]
                #from ipdb import set_trace
                #set_trace() 
                ###函数im_detect的输出是什么样的？这里为啥要乘上4？？？？？？？？？？？
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

                ##index的每一行的结构((start_x,start_y,end_x,end_y),h_num*(j-1)+k)
                # from ipdb import set_trace
                # set_trace()
                cls_dets[:,:1]=(cls_dets[:,:1]+index[im_index][0][1])
                cls_dets[:,1:2]=(cls_dets[:,1:2]+index[im_index][0][0])
                cls_dets[:,2:3]=(cls_dets[:,2:3]+index[im_index][0][1])
                cls_dets[:,3:4]=(cls_dets[:,3:4]+index[im_index][0][0])
                all_dets[1].append(cls_dets.tolist())
 
        # from ipdb import set_trace
        # set_trace() 
        for j in xrange(1, 2):
            if len(all_dets[j])==0:
               continue
            whole_dets=np.vstack(all_dets[j])  #这里结束得到的是单个机场裁完测的所有结果
        return whole_dets   

    else:
        scores, boxes = im_detect(net, im)
        # from ipdb import set_trace
        # set_trace()         
        for class_index in xrange(1, 2):
            #print(class_index)
            inds = np.where(scores[:, 1] > CONF_THRESH[0])[0]  #confidence thresh
            if len(inds)==0:
                continue
            #############################
            #print(inds)
            ###############################
            cls_scores = scores[inds, 1]
            cls_boxes = boxes[inds, 4:8]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) 
            return cls_dets

def demo_airport(net, im, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    ##im1是画图后保存的图片，首先为原图im,若检测船时已画图保存过则读取画过船的图，在上面接着画
    im1 = im 
    if os.path.exists(save_path+image_name):
        im1 = cv2.imread(save_path+image_name)
   
    ##检测机场
    scores, boxes = im_detect(net, im)

    ##对机场这一类的结果处理
    for class_index in xrange(3, 4):
            #print(class_index)
        inds = np.where(scores[:, 1] > CONF_THRESH[2])[0]  #confidence thresh
        if len(inds)==0:
            continue
            #############################
            #print(inds)
            ###############################
        cls_scores = scores[inds, 1]
        cls_boxes = boxes[inds, 4:8]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            # keep2=postprocess(cls_dets,del_theta,del_theta_p)
            # all_dets_pos=cls_dets[keep2]
            #keep = soft_nms(cls_dets, sigma=0.5, Nt=0.3, method=2, threshold=0.001) 
        keep = nms(cls_dets, NMS_THRESH[2]) #nms thresh
        cls_dets = cls_dets[keep]
            
            # ind=postprocess2(cls_dets,del_theta2[class_index-1])
            # cls_dets=cls_dets[ind]
            # a_arg=np.argsort(-cls_dets[:,4])
            # cls_dets=cls_dets[a_arg]


        ##对每个检测出来的机场再继续检测飞机
        dets = []
        for i in range(cls_dets.shape[0]):
            bbox = tuple(int(np.round(x)) for x in cls_dets[i, :4])
            score = cls_dets[i, -1]
            if score>0:
                cv2.rectangle(im1, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                cv2.putText(im1, '%s: %.3f' % (CLASSES[3], score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                         1.0, (0, 0, 255), thickness=1)
                img = im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                whole_dets = demo_plane(net3,img)
                whole_dets[:,:1]=(whole_dets[:,:1]+bbox[0])
                whole_dets[:,1:2]=(whole_dets[:,1:2]+bbox[1])
                whole_dets[:,2:3]=(whole_dets[:,2:3]+bbox[0])
                whole_dets[:,3:4]=(whole_dets[:,3:4]+bbox[1])
                dets.append(whole_dets)
        dets = np.array(dets)
        # from ipdb import set_trace
        # set_trace()
        if len(dets)==0:
            continue
        dets = np.vstack(dets[0])  #所有机场上飞机的检测结果
        dets = dets.astype(np.float32, copy=False)
        keep = nms(dets, NMS_THRESH[0]) 
        dets = dets[keep]

        if os.path.exists(result_path):
            pass
        else:
            os.mkdir(result_path)

        if image_name.find('.tiff') == -1:  # this img is png or tif
            txt_name=image_name[:-4]
        else:  #this img is tiff
            txt_name=image_name[:-5]

        file2=open(result_path+txt_name+'.txt','a') #写飞机检测结果的txt
        for i in range(dets.shape[0]):
            bbox = tuple(int(np.round(x)) for x in dets[i, :4])
            score = dets[i, -1]
            if score>0.5:
                cv2.rectangle(im1, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                cv2.putText(im1, '%s: %.3f' % (CLASSES[1], score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                         1.0, (0, 0, 255), thickness=1)
               
            line=CLASSES[1]+' '+str(score)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n'
            file2.write(line)
        file2.close()
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)        
    cv2.imwrite(os.path.join(save_path+'/'+image_name),im1)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)#默认使用gpu 0
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    if not os.path.isfile(caffemodel_airport):
        raise IOError(('{:s} not found.').format(caffemodel_airport))
    if not os.path.isfile(caffemodel_plane):
        raise IOError(('{:s} not found.').format(caffemodel_plane))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net2 = caffe.Net(prototxt_airport, caffemodel_airport, caffe.TEST)
    net3 = caffe.Net(prototxt_plane, caffemodel_plane, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel_airport)
    print '\n\nLoaded network {:s}'.format(caffemodel_plane)
    
    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
        # _, _= im_detect(net, im)
    
    time = 0
    im_names = os.listdir(im_path)
    num = len(im_names)
    # num=len(filenames)
    for i,im_name in enumerate(im_names):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'This is the image', i, '/',num,':', im_name
        im_file = os.path.join(im_path, im_name)
        timer = Timer()
        timer.tic()
        timer.tic()
        im = cv2.imread(im_file)
        timer.toc()
        print ('reading image took {:.3f}s for detection').format(timer.total_time) 
        ##然后检测机场
        demo_airport(net2, im, im_name)
        timer.toc()
        print ('Detection took {:.3f}s for detection').format(timer.total_time)
        global time
        time += timer.total_time
    print 'Average time = ',time/num

