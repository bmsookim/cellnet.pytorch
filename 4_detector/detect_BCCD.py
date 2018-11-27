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
from detect_cell import softmax, getNetwork, return_class_idx, generate_sliding_windows, check_and_mkdir

if not sys.warnoptions:
    import warnings
    warning.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--stepSize', default=80, type=int, help='size of each sliding window steps')
parser.add_argument('--windowSize', default=240, type=int, help='size of the sliding window')
parser.add_argument('--subtype', default=None, type=str, help='Type to find')
parser.add_argument('--testNumber', default=1, type=int, help='Test Number')
args = parser.parse_args()


in_size = 224 if args.net_type == 'resnet' else 299

# Phase 1 : Model Upload
print("\n[Phase 1] : Model Weight Upload")
use_gpu = torch.cuda.is_available()

data_dir = cf.aug_base
trainset_dir = cf.data_base.split("/")[-1]+os.sep

dsets = datasets.ImageFolder(data_dir, None)
H = datasets.ImageFolder(os.path.join(cf.aug_base, 'train'))
dset_classes = H.classes

if __name__ == "__main__":
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

    sample_input = Variable(torch.randn(1,3,224,224), volatile=False)
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

    file_name = cf.test_dir + os.sep + ("")
