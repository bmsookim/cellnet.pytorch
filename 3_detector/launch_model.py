# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Description : inference.py
# The main code for inference test phase of trained model.
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys
import argparse
import csv
import operator

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
args = parser.parse_args()

# Phase 1 : Model Upload
print('\n[Phase 1] : Model Weight Upload')
use_gpu = torch.cuda.is_available()

# upload labels
data_dir = cf.test_base
trainset_dir = cf.data_base.split("/")[-1]+os.sep

dsets = datasets.ImageFolder(data_dir, None)
H = datasets.ImageFolder(os.path.join(cf.aug_base, 'train'))
dset_classes = H.classes

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'resnet'):
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('[Error]: Network should be either [alexnet / vgget / resnet]')
        sys.exit(1)

    return file_name

# uploading the model
print("| Loading checkpoint model for grad-CAM...")
assert os.path.isdir('../4_classifier/checkpoint'),'[Error]: No checkpoint directory found!'
assert os.path.isdir('../4_classifier/checkpoint/'+trainset_dir),'[Error]: There is no model weight to upload!'
file_name = getNetwork(args)
checkpoint = torch.load('../4_classifier/checkpoint/'+trainset_dir+file_name+'.t7')
model = checkpoint['model']

if use_gpu:
    model.cuda()
    cudnn.benchmark = True

model.eval()

sample_input = Variable(torch.randn(1,3,224,224), volatile=True)
if use_gpu:
    sampe_input = sample_input.cuda()

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean, cf.std)
])

# Inference testout, Upload a single image
img = Image.open('/home/bumsoo/Data/test/CELL_PATCHES/WBC_Neutrophil_Band/WBC_B150.png')
if test_transform is not None:
    img = test_transform(img)
inputs = img
inputs = Variable(inputs, volatile=True)

if use_gpu:
    inputs = inputs.cuda()
inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

outputs = model(inputs)
softmax_res = softmax(outputs.data.cpu().numpy()[0])

index,score = max(enumerate(softmax_res), key=operator.itemgetter(1))

print('prediction = ' + dset_classes[index])
