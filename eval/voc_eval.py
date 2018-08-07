# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import os
import cPickle
import numpy as np
import pdb

class_dict = {2:'ship'}
def parse_rec_remote(filename):
    objs = open(filename,'r').readlines()
    objects = []
    for obj in objs:
        if len(obj)<=2:
            continue
        content = obj[:-2].split(' ')
        obj_struct = {}
        obj_struct['name'] = class_dict[int(content[1])]
        obj_struct['pose'] = 0
        obj_struct['truncated'] = 0
        obj_struct['difficult'] = 0
        bbox = content[2:]
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[2])#+x1
        y2 = float(bbox[3])#+y1
        obj_struct['bbox'] = [x1, y1, x2, y2]
        objects.append(obj_struct)

    return objects
    
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def remote_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.7,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annotstest.pkl')
    imagenames = imagesetfile
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec_remote(annopath.format(imagename[:-5]))
            ##recs里面的数据形式如下：{'Yangy22_2_20.jpg': [{'difficult': 0, 'pose': 0, 'name': 'vessle', 'bbox': [100.0, 16.0, 158.0, 67.0], 'truncated': 0}, {'difficult': 0, 'pose': 0, 'name': 'vessle', 'bbox': [3.0, 102.0, 75.0, 174.0], 'truncated': 0}, {'difficult': 0, 'pose': 0, 'name': 'vessle', 'bbox': [100.0, 16.0, 158.0, 67.0], 'truncated': 0}, {'difficult': 0, 'pose': 0, 'name': 'vessle', 'bbox': [3.0, 102.0, 75.0, 174.0], 'truncated': 0}, {'difficult': 0, 'pose': 0, 'name': 'vessle', 'bbox': [124.0, 0.0, 179.0, 35.0], 'truncated': 0}, {'difficult': 0, 'pose': 0, 'name': 'vessle', 'bbox': [124.0, 0.0, 179.0, 35.0], 'truncated': 0}]}
            ##len(recs)标志有目标的图像的数目
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)
    
    # extract gt objects for this class ######处理图像的真实标注结果
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        #imagename='wangqi109_2_3.jpg'
        #pdb.set_trace()
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    ##########没有标注类似如下{'det': [], 'bbox': array([], dtype=float64), 'difficult': array([], dtype=bool)}
    ###########有标注类似如下：{'det': [False, False, False, False, False, False, False, False], 'bbox': array([[  46.,   99.,  135.,  220.],
       #[ 126.,  190.,  225.,  302.],
       #[ 321.,  303.,  422.,  434.],
       #[ 445.,  290.,  555.,  438.],
       #[  46.,   99.,  135.,  220.],
       #[ 126.,  190.,  225.,  302.],
       #[ 321.,  303.,  422.,  434.],
       #[ 445.,  290.,  555.,  438.]]), 'difficult': array([False, False, False, False, False, False, False, False], dtype=bool)}
    # read dets ##########处理我们模型测出来的检测结果
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:  ########any(lines)判断lines是否为空，若是则为false

        splitlines = [x.strip().split(' ') for x in lines] ####这个是把所有的结果写在了splitlines里面。形式如： [['licongying068_2_2.jpg', '0.9', '7', '14', '148', '144'], ['licongying068_2_2.jpg', '0.9', '169', '272', '263', '379']]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])#####表示测试得到的所有label的得分
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])#####表示测试得到的所有label的坐标
        ######测到的目标是多少，splitlines，confidence，image_ids，BB的长度就是多少
        # sort by confidence 根据得分，将测试得到的label进行了排序
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        # go down dets and mark TPs and FPs
        #######这个代码是以我们的检测结果为基础的，也就是说拿到一个检测结果，对应着找检测结果所在图的真实标注。判断检测结果是否真确
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd): ######这个循环的是我的所有检测结果
            R = class_recs[image_ids[d]] ####R里面是检测结果对应的图中的所有真实标注
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                
                # intersection计算检测结果与真实标注的交
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.) #####相交区域的宽高
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih ######相交区域的面积

                # union计算检测结果与真实标注的并
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni ##计算IOU值
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        ######
        TT=sum(tp)/float(npos)
        FF=sum(fp)/float(npos)
        print ".....................ture or false",TT,FF
        #######
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
         rec = -1
         prec = -1
         ap = -1

    return rec, prec, ap