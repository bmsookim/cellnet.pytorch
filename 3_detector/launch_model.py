# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Grad CAM Implementation
#
# Description : launch_model.py
# The main code for grad-CAM image localization.
# ***********************************************************

from __future__ import print_function, division

import cv2
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
from grad_cam import *

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image
from misc_functions import save_class_activation_on_image
from grad_cam import BackPropagation, GradCAM, GuidedBackPropagation
from guided_backprop import GuidedBackprop

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

sample_input = Variable(torch.randn(1,3,224,224), volatile=False)
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

"""
#@ Code for inference test

img = Image.open(cf.image_path)
if test_transform is not None:
    img = test_transform(img)
inputs = img
inputs = Variable(inputs, volatile=False, requires_grad=True)

if use_gpu:
    inputs = inputs.cuda()
inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

outputs = model(inputs)
softmax_res = softmax(outputs.data.cpu().numpy()[0])

index,score = max(enumerate(softmax_res), key=operator.itemgetter(1))

print('| Uploading %s' %(cf.image_path.split("/")[-1]))
print('| prediction = ' + dset_classes[index])
"""

#@ Code for extracting a grad-CAM region for a given class
gcam = GradCAM(model._modules.items()[0][1], cuda=use_gpu)#model=model._modules.items()[0][1], cuda=use_gpu)
gbp = GuidedBackPropagation(model=model._modules.items()[0][1], cuda=use_gpu)

print(dset_classes)
# WBC_id = 16 # Neutrophil Segmented
WBC_id = 3 # Eosinophil
print("Checking Activated Regions for " + dset_classes[WBC_id] + "...")

for i in range(14):
    file_name = './cell_data/2_%s_Neutrophil Segmented.png' %(str(i))
    print(file_name)

    original_image = cv2.imread(file_name)
    img = Image.open(file_name)
    if test_transform is not None:
        img = test_transform(img)
        original_image = cv2.resize(original_image, (256, 256))

    center_cropped = original_image[16:240, 16:240, :]
    inputs = img
    inputs = Variable(inputs, requires_grad=True)

    if use_gpu:
        inputs = inputs.cuda()
    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

    probs, idx = gcam.forward(inputs)
    #probs, idx = gbp.forward(inputs)

    # Grad-CAM
    gcam.backward(idx=WBC_id)
    output = gcam.generate(target_layer='layer4.2')

    # Guided Back Propagation
    #gbp.backward(idx=WBC_id)
    #feature = gbp.generate(target_layer='conv1')

    # Guided Grad-CAM
    #output = np.multiply(feature, region)

    """
    """
    gcam.save('./results/%s.png' %str(i), output, center_cropped)
    cv2.imwrite('./results/map%s.png' %str(i), output*255)

    for j in range(3):
        print('\t{:5f}\t{}\n'.format(probs[j], dset_classes[idx[j]]))

"""
@ Code for extracting the Top-3 Results for each image
topk = 3

for i in range(0, topk):
    gcam.backward(idx=idx[i])
    output = gcam.generate(target_layer='layer4.2')

    gcam.save('./results/{}_gcam.png'.format(dset_classes[idx[i]]), output, center_cropped)
    print('\t{:.5f}\t{}'.format(probs[i], dset_classes[idx[i]]))
"""
