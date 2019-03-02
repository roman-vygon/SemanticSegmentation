from skimage.io import imread, imshow, imsave
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import os, sys, glob
from collections import namedtuple
import fnmatch


classes = ['road'   ,
'sidewalk',
'building' ,
'wall'      ,
'fence'      ,
'pole'        ,
'traffic light',
'traffic sign'  ,
'vegetation'    ,
'terrain'       ,
'sky'           ,
'person'        ,
'rider'         ,
'car'           ,
'truck'         ,
'bus'           ,
'train'         ,
'motorcycle'    ,
'bicycle'    ]

def eval_preds(preds, groundTruth):
	"""
	Counts IoU score for predicted images
	"""
    eps = 1e-8
    class_ious = np.zeros(20)
    iou_score = 0
    for i in range(preds.shape[0]):
        sys.stdout.write('\r')
        sys.stdout.write("processed %d images" % i)
        sys.stdout.flush()
        pred = preds[i]
        gt = groundTruth[i]
        for j in range(20):
            pred1 = (pred == j)
            gt1 = (gt == j)
            intersect = np.logical_and(pred1, gt1).sum()
            union = np.logical_or(pred1, gt1).sum()
            if union != 0:
                class_ious[j]+=(intersect)/(union)
            else:
                class_ious[j]+=1
    class_ious/=preds.shape[0]
    print()
    for i in range(len(classes)):
        print(classes[i],':',class_ious[i])
    return np.mean(class_ious)



