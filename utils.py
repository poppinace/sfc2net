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


#def rsquared(pd, gt):
#    """ Return R^2 where x and y are array-like."""
#    pd, gt = np.array(pd), np.array(gt)
#    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
#    return r_value**2
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


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    new_x = cv.resize(x, dsize=(w,h), interpolation=cv.INTER_CUBIC)

    return new_x

def dense_sample2d(x, sx, stride):
    (h,w) = x.shape[:2]
    #idx_img = np.array([i for i in range(h*w)]).reshape(h,w)
    idx_img = np.zeros((h,w),dtype=float)
    
    th = [i for i in range(0, h-sx+1, stride)]
    tw = [j for j in range(0, w-sx+1, stride)]
    norm_vec = np.zeros(len(th)*len(tw))
    
    
    for i in th:
        for j in tw:
            idx_img[i:i+sx,j:j+sx] = idx_img[i:i+sx,j:j+sx]+1
   
    idx_img = 1/idx_img
    idx_img = idx_img/sx/sx
    #line order
    idx = 0
    for i in th:
        for j in tw:
            norm_vec[idx] =idx_img[i:i+sx,j:j+sx].sum()
            idx+=1
    
    return norm_vec

def upsample_countmap(pred, image, patch_sz, stride):
    pred = pred.reshape(-1)
    imH, imW = image.shape[2:4]
    cntMap = np.zeros((imH, imW), dtype=float)
    
    H = np.arange(0, imH - stride + 1, stride)
    W = np.arange(0, imW - stride + 1, stride)
    cnt = 0
    for h in H:
        for w in W:
            pixel_cnt = pred[cnt] / stride / stride
            cntMap[h:h+stride, w:w+stride] += pixel_cnt
            cnt += 1
    return cntMap

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


def recover_countmapp(pred, image, patch_sz, stride):
    pred = pred.reshape(-1)
    imH, imW = image.shape[1:3]
    cntMap = np.zeros((imH, imW), dtype=float)
    norMap = np.zeros((imH, imW), dtype=float)
    
    H = np.arange(0, imH - patch_sz + 1, stride)
    W = np.arange(0, imW - patch_sz + 1, stride)
    cnt = 0
    for i in range(len(H)):
        for j in range(len(W)):
            pixel_cnt = pred[cnt] / patch_sz / patch_sz
            cntMap[H[i]:H[i]+patch_sz, W[j]:W[j]+patch_sz] += pixel_cnt
            norMap[H[i]:H[i]+patch_sz, W[j]:W[j]+patch_sz] += 1
            cnt += 1
    return cntMap / (norMap + 1e-12)


# def data_generator(data_dir='data/Train400', verbose=False):
#     # generate clean patches from a dataset
#     file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
#     # initrialize
#     data = []
#     # generate patches
#     for i in range(len(file_list)):
#         patches = gen_patches(file_list[i])
#         for patch in patches:    
#             data.append(patch)
#         if verbose:
#             print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
#     data = np.array(data, dtype='uint8')
#     data = np.expand_dims(data, axis=3)
#     discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
#     data = np.delete(data, range(discard_n), axis=0)
#     print('^_^-training data finished-^_^')
#     return data