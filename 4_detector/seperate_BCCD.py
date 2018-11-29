from __future__ import print_function, division

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import time
import copy
import os
import sys
import argparse
import csv
import operator
import pickle
import progressbar
import grad_cam
import networks
import BCCD_config as cf

from functools import partial
from time import sleep
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from misc_functions import save_class_activation_on_image
from grad_cam import BackPropagation, GradCAM, GuidedBackPropagation
from detect_cell import softmax, getNetwork, return_class_idx, check_and_mkdir, generate_sliding_windows

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--stepSize', default=80, type=int, help='size of each sliding window steps')
parser.add_argument('--windowSize', default=120, type=int, help='size of the sliding window')
parser.add_argument('--subtype', default=None, type=str, help='Type to find')
parser.add_argument('--testNumber', default=1, type=int, help='Test Number')
args = parser.parse_args()

in_size = 224 if args.net_type == 'resnet' else 299

# Phase 1 : Model Upload
use_gpu = torch.cuda.is_available()

data_dir = cf.aug_base
trainset_dir = cf.data_base.split("/")[-1]+os.sep

dsets = datasets.ImageFolder(data_dir, None)
H = datasets.ImageFolder(os.path.join(cf.aug_base, 'train'))
dset_classes = H.classes

if __name__ == "__main__":
    try:
        xrange
    except NameError:
        xrange = range

    print("| Loading checkpoint model for grad-CAM...")
    assert os.path.isdir('../3_classifier/checkpoint'),'[Error]: No checkpoint directory found!'
    assert os.path.isdir('../3_classifier/checkpoint/'+trainset_dir),'[Error]: There is no model weight to upload!'
    file_name = getNetwork(args)
    if (sys.version_info > (3,0)):
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        checkpoint = torch.load('../3_classifier/checkpoint/'+trainset_dir+file_name+'.t7', pickle_module=pickle)
    else:
        checkpoint = torch.load('../3_classifier/checkpoint/'+trainset_dir+file_name+'.t7')
    model = checkpoint['model']

    if use_gpu:
        model.cuda()
        cudnn.benchmark = True

    model.eval()

    sample_input = Variable(torch.randn(1,3,in_size,in_size), volatile=False)
    if use_gpu:
        sampe_input = sample_input.cuda()

    def is_image(f):
        return f.endswith(".png") or f.endswith(".jpg")

    test_transform = transforms.Compose([
        transforms.Scale(in_size),
        transforms.CenterCrop(in_size),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])

    gcam = GradCAM(list(model._modules.items())[0][1], cuda=use_gpu)

    print("\n[Phase 2] : Gradient Detection")
    if args.subtype != None:
        WBC_id = return_class_idx(args.subtype)

        if not (args.subtype in dset_classes):
            print("The given subtype does not exists!")
            sys.exit(1)

    if args.subtype == None:
        print("| Checking All Activated Regions...")
    else:
        print("| Checking Activated Regions for " + dset_classes[WBC_id] + "...")

    for subdir, dirs, files in os.walk(cf.test_base):
        for f in files:
            file_name = os.path.join(subdir, f)
            if not is_image(f):
                continue
            print("| Opening "+file_name+"...")

            image = Image.open(file_name)
            img_lst = [image.crop((0,0,480,480)), image.crop((160,0,640,480))]

            background_img = cv2.imread(file_name)
            blank_mask = np.zeros((background_img.shape[0], background_img.shape[1]))
            blank_heatmap = np.zeros((background_img.shape[0], background_img.shape[1]))

            mode = 0
            for img in img_lst:
                if test_transform is not None:
                    image = test_transform(img)
                inputs = image

                with torch.no_grad():
                    inputs = Variable(inputs)

                if use_gpu:
                    inputs = inputs.cuda()

                inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                probs, idx = gcam.forward(inputs)

                if (args.subtype == None):
                    comp_idx = idx[0]
                    item_id = 0
                else:
                    comp_idx = WBC_id
                    item_id = (np.where(idx.cpu().numpy() == (WBC_id)))[0][0]

                gcam.backward(idx=comp_idx)
                output = gcam.generate(target_layer = 'layer4.2') # for resnet

                heatmap = output
                original = inputs.data.cpu().numpy()

                original = np.transpose(original, (0,2,3,1))[0]
                original = original * cf.std + cf.mean
                original = np.uint8(original * 255.0)

                mask = np.uint8(heatmap * 255.0)

                blank_heatmap[:, (mode*160):480+(mode*160)] = cv2.resize(heatmap, (480, 480))
                mode += 1

            blank_heatmap[blank_heatmap > 1] = 1
            blank_heatmap = cv2.GaussianBlur(blank_heatmap, (15, 15), 0)
            blank_mask = np.uint8(blank_heatmap * 255.0)
            check_and_mkdir("./results/BCCD/heatmaps")
            check_and_mkdir("./results/BCCD/masks")

            save_dir = "./results/BCCD/heatmaps/" + f
            mask_dir = "./results/BCCD/masks/" + f

            gcam.save(save_dir, blank_heatmap, background_img)
            cv2.imwrite(mask_dir, blank_mask)
