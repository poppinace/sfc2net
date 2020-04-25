"""
@author: hao
"""

import scipy
import numpy as np
import math
import cv2 as cv
from scipy.ndimage import gaussian_filter, morphology
#import scipy.stats.linregress 
#from skimage.measure import label, regionprops


def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae


def compute_mse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse

def compute_relerr(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        relerr = np.mean(np.abs(diff) / gt * 100)
    else:
        relerr = 0

    diff = diff[gt > 10]
    gt = gt[gt > 10]
    if (diff.size != 0) and (gt.size != 0):
        relerr10 = np.mean(np.abs(diff) / gt * 100)
    else:
        relerr10 = float('NaN')
    return relerr, relerr10

def rsquared(x,y):
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    SSR=0
    Varx=0
    Vary=0
    for i in range(0,len(x)):
        SSR+=(x[i]-x_bar)*(y[i]-y_bar)
        Varx+=(x[i]-x_bar)**2
        Vary+=(y[i]-y_bar)**2
    SST=math.sqrt(Varx*Vary)
    return (SSR/SST)**2

def recover_countmap(pred, image, patch_sz, stride):
    pred = pred.reshape(-1)
    imH, imW = image.shape[2:4]
    cntMap = np.zeros((imH, imW), dtype=float)
    norMap = np.zeros((imH, imW), dtype=float)
    
    H = np.arange(0, imH - patch_sz + 1, stride)
    W = np.arange(0, imW - patch_sz + 1, stride)
    cnt = 0
    for h in H:
        for w in W:
            pixel_cnt = pred[cnt] / patch_sz / patch_sz
            cntMap[h:h+patch_sz, w:w+patch_sz] += pixel_cnt
            norMap[h:h+patch_sz, w:w+patch_sz] += np.ones((patch_sz,patch_sz))
            cnt += 1
    return cntMap / (norMap + 1e-12)
