# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import cPickle
import math
import glob
import uuid
import scipy.io as sio
import pdb
from imdb import imdb
from imdb import ROOT_DIR
#import ds_utils
from voc_eval import remote_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project



# <<<< obsolete



class remote_data(imdb):
    def __init__(self,pascal_path='/share/manage/NEU/home3/work/pytorch/detection/new_remote_data'):
        imdb.__init__(self, 'newremotedata')
        self._image_set = 'trainval'
        self._pascal_path = pascal_path
        self._data_path = os.path.join(self._pascal_path, 'img')
        self._classes = ('__background__', # always index 0
                         'ship')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        #image_path = os.path.join(self._data_path, 'JPEGImages',
        #                          index + self._image_ext)
        image_path = os.path.join(self._data_path, index[:-4]+self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._pascal_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        #image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
        #                              self._image_set + '.txt')
        #assert os.path.exists(image_set_file), \
        #        'Path does not exist: {}'.format(image_set_file)
        #with open(image_set_file) as f:
        #    image_index = [x.strip() for x in f.readlines()]
        image_index = os.listdir(self._data_path)
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        #return os.path.join(ROOT_DIR, 'data', 'PASCAL')
        return os.path.join(ROOT_DIR, 'new_remote_data')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
                                    
    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join('results', self.name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
    
    def _do_python_eval(self,output_dir,anno,imageset):
        annopath = os.path.join(anno,'{:s}.txt') ##需要评价的所有图像的真实标注
        imagesetfile = os.listdir(imageset) ##需要评价的所有图像的名字
        cachedir = os.path.join('results','annotations_cache') ##评价代码results/annotations_cache'的路径
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False #if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
        ######('__background__', 'aeroplane', 'vessle')为self._classes的内容
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)##这个是我们自己测到的结果存储的路径
            #pdb.set_trace()
            rec, prec, ap = remote_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, output_dir,anno,imageset):
        self._do_python_eval(output_dir,anno,imageset)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
