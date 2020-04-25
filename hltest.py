"""
@author: hao
"""

import os
import argparse
from time import time

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from hlnet import *
from hldatasetv2 import *
from utils import *
import torchvision.models as models
cudnn.enabled = True

# constant
IMG_SCALE = 1./255
IMG_MEAN = [ 0.4607, 0.4836, 0.3326 ]
IMG_STD = [0.1058, 0.1065, 0.0999 ]

# system-related parameters
DATA_DIR = './data/rice_datasets-test/'
DATASET = 'rice'
EXP = 'sfc2net'
DATA_VAL_LIST = './data/rice_datasets-test/test.txt'

RESTORE_FROM = 'model_best.pth.tar'
SNAPSHOT_DIR = './snapshots'
RESULT_DIR = './results'

# classification-related parameters
STEP_DEFAULT=0.1
MAX_CLASS_NUMBER=80
START_DEFAULT=-2

# model-related parameters
WIDTH_MULT = 1.0
INPUT_SIZE = 32
OUTPUT_STRIDE = 8
MODEL = 'mixnet_fusion_classification'
RESIZE_RATIO = 0.25

# training-related parameters
NUM_CPU_WORKERS = 0
PRINT_EVERY = 1
RANDOM_SEED = 6
VAL_EVERY = 10

#dataset parameters
GAUSS_KERNEL=4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Counting Framework")
    # constant
    parser.add_argument("--image-scale", type=float, default=IMG_SCALE, help="Scale factor used in normalization.")
    parser.add_argument("--image-mean", nargs='+', type=float, default=IMG_MEAN, help="Mean used in normalization.")
    parser.add_argument("--image-std", nargs='+', type=float, default=IMG_STD, help="Std used in normalization.")
    # system-related parameters
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset type.")
    parser.add_argument("--exp", type=str, default=EXP, help="Experiment path.")
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST, help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Name of restored model.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR, help="Where to save inferred results.")
    parser.add_argument("--save-output", action="store_true", help="Whether to save the output.")
    # model-related parameters
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE, help="the minimum input size of the model.")
    parser.add_argument("--output-stride", type=int, default=OUTPUT_STRIDE, help="Output stride of the model.")
    parser.add_argument("--resize-ratio", type=float, default=RESIZE_RATIO, help="Resizing ratio.")
    parser.add_argument("--model", type=str, default=MODEL, help="model to be chosen.")
    parser.add_argument("--width-mult", type=float, default=WIDTH_MULT, help="Decoder kernel size.")
    parser.add_argument("--use-pretrained", action="store_true", help="Whether to use pretrained model.")
    parser.add_argument("--freeze-bn", action="store_true", help="Whether to freeze encoder bnorm layers.")
    parser.add_argument("--sync-bn", action="store_true", help="Whether to apply synchronized batch normalization.")
    parser.add_argument("--use-nonlinear", action="store_true", help="Whether to use nonlinearity in IndexNet.")
    parser.add_argument("--use-context", action="store_true", help="Whether to use context in IndexNet.")
    parser.add_argument("--use-squeeze", action="store_true", help="Whether to squeeze IndexNet.")
    # training-related parameters
    parser.add_argument("--evaluate-only", action="store_true", help="Whether to perform evaluation.")
    parser.add_argument("--num-workers", type=int, default=NUM_CPU_WORKERS, help="Number of CPU cores used.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY, help="Print information every often.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results.")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY, help="How often performing validation.")
    # classification parameters
    parser.add_argument("--step-log", type=float, default=STEP_DEFAULT, help="Quantization step in the log space.")
    parser.add_argument("--max-class-number", type=int, default=MAX_CLASS_NUMBER, help="Max class number.")
    parser.add_argument("--start-log", type=int, default=START_DEFAULT, help="Start number in the log space.")
    # dataset parameters
    parser.add_argument("--gauss-kernel", type=int, default=GAUSS_KERNEL, help="Max class number.")
    
    return parser.parse_args()

def test(net, testset, test_loader, args):
    # switch to 'eval' mode
    net.eval()
    cudnn.benchmark = False
    
    # image_list = valset.image_list
    image_list = [name.split('\t') for name in open(args.data_val_list).read().splitlines()]
      
    epoch_result_dir = os.path.join(args.result_dir, str(0))
    if not os.path.exists(epoch_result_dir):
        os.makedirs(epoch_result_dir)
    cmap = plt.cm.get_cmap('jet')

    pd_counts = []
    gt_counts = []
    with torch.no_grad():
        avg_frame_rate = 0.0
        for i, sample in enumerate(test_loader):            
            torch.cuda.synchronize()
            start = time()

            image, gtcount = sample['image'], sample['gtcount']
            h = image.shape[2]
            w = image.shape[3]
                                          
            ht=int(32*int(h/32))    
            wt=int(32*int(w/32))
            if ht!=h:
                ht=int(32*(int(h/32)+1))  
            if wt!=w:
                wt=int(32*(int(w/32)+1))  
                                
            Img_t=torch.FloatTensor(np.zeros((1,3,ht,wt)))
            Img_t[:,:,0:h,0:w]=image
            image=Img_t
                
            # inference
            output,countmap = net(image.cuda())
            output = np.clip(output.squeeze().cpu().numpy(), 0, None)
            countmap = np.clip(countmap.squeeze().cpu().numpy(), 0, None)
            
            pdcount = output.sum()
            gtcount = float(gtcount.numpy())

            if True:
                # image_name = image_list[i]
                _, image_name = os.path.split(image_list[i][0])
                output_save = recover_countmap(countmap, image, args.input_size, args.output_stride)
                output_save = output_save / (output_save.max() + 1e-12)
                output_save = cmap(output_save) * 255.
                # image composition
                # image = valset.images[image_name]
                image = testset.images[image_list[i][0]]
                nh, nw = output_save.shape[:2]
                image = cv2.resize(image, (nw, nh), interpolation = cv2.INTER_CUBIC)
                output_save = 0.5 * image + 0.5 * output_save[:, :, 0:3]

                fig = plt.imshow(output_save.astype(np.uint8))
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.title('pred=%4.2f, gt=%4.2f'%(pdcount, gtcount), {'fontsize':8}), plt.xticks([]), plt.yticks([])
                plt.savefig(os.path.join(epoch_result_dir, image_name.replace('.jpg', '.png')), bbox_inches='tight', dpi = 300)
                plt.close()
                
            # compute mae and mse
            pd_counts.append(pdcount)
            gt_counts.append(gtcount)
            mae = compute_mae(pd_counts, gt_counts)
            mse = compute_mse(pd_counts, gt_counts)
            relerr, relerr10 = compute_relerr(pd_counts, gt_counts)

            torch.cuda.synchronize()
            end = time()
            if end - start==0:
                running_frame_rate=999
            else:
                running_frame_rate = float(1 / (end - start))
            avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
            if i % args.print_every == args.print_every - 1:
                print(
                    'epoch: {0}, test: {1}/{2}, pre: {3:.2f}, gt:{4:.2f}, me:{5:.2f}, mae: {6:.2f}, mse: {7:.2f}, relerr: {8:.2f}%, relerr10: {9:.2f}%, frame: {10:.2f}Hz/{11:.2f}Hz'
                    .format(0, i+1, len(test_loader), pdcount, gtcount, pdcount-gtcount, mae, mse, relerr, relerr10, running_frame_rate, avg_frame_rate)
                )
            start = time()
    r2 = rsquared(pd_counts, gt_counts)
    np.save(r"pd_counts_tcsvt.npy",pd_counts)
    np.save(r"gt_counts_tcsvt.npy",gt_counts)
    print('epoch: {0} mae: {1:.2f}, mse: {2:.2f}, relerr: {3:.2f}%, relerr10: {4:.2f}%, r2: {5:.2f}'.format(0, mae, mse, relerr, relerr10, r2))

def main():
    args = get_arguments()

    args.image_mean = np.array(args.image_mean).reshape((1, 1, 3))
    args.image_std = np.array(args.image_std).reshape((1, 1, 3))
    
    # seeding for reproducbility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # instantiate dataset
    dataset_list = {
        'rice': RiceDataset
    }
    args.evaluate_only=True
    dataset = dataset_list[args.dataset]
    
    args.snapshot_dir = os.path.join(args.snapshot_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    args.result_dir = os.path.join(args.result_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.restore_from = os.path.join(args.snapshot_dir, args.restore_from)

    arguments = vars(args)
    for item in arguments:
        print(item, ':\t' , arguments[item])
    
    # filter parameters   
    net= Mixnet_l_backbone(pretrained=args.use_pretrained,model_name=args.model,\
                           max_class_number=args.max_class_number,step_log=args.step_log, start_log=args.start_log,\
                           input_size=args.input_size,output_stride=args.output_stride,freeze_bn=args.freeze_bn)
    net.cuda()

    if args.restore_from is not None:
        if os.path.isfile(args.restore_from):
            checkpoint = torch.load(args.restore_from)
            net.load_state_dict(checkpoint['state_dict'])
            print("==> load checkpoint '{}'"
                  .format(args.restore_from))
        else:
            with open(os.path.join(args.snapshot_dir, args.exp+'.txt'), 'a') as f:
                for item in arguments:
                    print(item, ':\t' , arguments[item], file=f)
            print("==> no checkpoint found at '{}'".format(args.restore_from))

    # define transform
    transform_val = [
        Normalize(
            args.image_scale, 
            args.image_mean, 
            args.image_std
        ),
        ToTensor(),
        ZeroPadding(args.output_stride)
    ]
    composed_transform_val = transforms.Compose(transform_val)

    testset = dataset(
        data_dir=args.data_dir,
        data_list=args.data_val_list,
        ratio=args.resize_ratio,
        train=False,
        transform=composed_transform_val,
        gauss_kernel=args.gauss_kernel
    )
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test(net, testset, test_loader, args)
    return
    
if __name__ == "__main__":
    main()
