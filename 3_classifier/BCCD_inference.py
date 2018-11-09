# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Description : BCCD_inference.py
# The main code for BCCD inference test phase of trained model.
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import BCCD_config as cf
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

parser = argparse.ArgumentParser(description='Pytorch Cell Classifier Training')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
args = parser.parse_args()

# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')

data_dir = cf.test_base
print("| Preparing %s dataset..." %(cf.test_base.split("/")[-1]))

use_gpu = torch.cuda.is_available()

# Set the classes of H1 labels
H = datasets.ImageFolder(os.path.join(cf.aug_dir+cf.H1_name, 'train'))
H_classes = H.classes

# Set the classes of Granulocytes labels
G = datasets.ImageFolder(os.path.join(cf.aug_dir+cf.G_name, 'train'))
G_classes = G.classes

# Set the classes of Mononuclear cells labels
M = datasets.ImageFolder(os.path.join(cf.aug_dir+cf.M_name, 'train'))
M_classes = M.classes

print("| Inferencing for %d classes" %len(H_classes))

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'densenet'):
        file_name = 'densenet-%s' %(args.depth)
    elif (args.net_type == 'resnet'):
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('[Error]: Network should be either [alexnet / vggnet / resnet]')
        sys.exit(1)

    return file_name

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print("| Loading checkpoint model for inference phase...")
assert os.path.isdir('checkpoint'), '[Error]: No checkpoint directory found!'
assert os.path.isdir('checkpoint/'+cf.H1_name), '[Error]: No model has been trained on Hierarchy #1 !'
file_name = getNetwork(args)
checkpoint = torch.load('./checkpoint/'+cf.H1_name+'/'+file_name+'.t7')
model = checkpoint['model']

checkpoint_G = torch.load('./checkpoint/'+cf.G_name+'/'+file_name+'.t7')
model_G = checkpoint_G['model']

checkpoint_M = torch.load('./checkpoint/'+cf.M_name+'/'+file_name+'.t7')
model_M = checkpoint_M['model']

# Hiearchical inference

if use_gpu:
    model.cuda()
    model_G.cuda()
    model_M.cuda()
    cudnn.benchmark = True

model.eval()
model_G.eval()
model_M.eval()

sample_input = Variable(torch.randn(1,3,224,224))
if use_gpu:
    sample_input = sample_input.cuda()

print("\n[Phase 3] : Score Inference")

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

# H1 Transform
H1_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean_H, cf.std_H)
])

# G Transform
G_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean_G, cf.std_G)
])

# M Transform
M_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean_M, cf.std_M)
])

if not os.path.isdir('result'):
    os.mkdir('result')

output_file = "./result/"+cf.test_base.split("/")[-1]+"_inference.csv"

errors = 0
with open(output_file, 'wb') as csvfile:
    fields = ['file_name', 'prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    cor = 0
    tot = 0

    for subdir, dirs, files in os.walk(data_dir):
        print(data_dir)
        for f in files:
            file_path = subdir + os.sep + f
            if (is_image(f)):
                tot += 1
                org_image = Image.open(file_path)#.convert('RGB')
                if H1_transform is not None:
                    image = H1_transform(org_image)
                else:
                    image = org_image
                inputs = image
                with torch.no_grad():
                    inputs = Variable(inputs)
                if use_gpu:
                    inputs = inputs.cuda()
                inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front

                outputs = model(inputs)
                softmax_res = softmax(outputs.data.cpu().numpy()[0])
                index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

                inf_class = H_classes[index]
                # Ground Truth
                inp = f.split("_")[0]

                if inf_class == 'Granulocytes':
                    if G_transform is not None:
                        image = G_transform(org_image)
                    else:
                        image = org_image
                    inputs = image
                    with torch.no_grad():
                        inputs = Variable(inputs)

                    if use_gpu:
                        inputs = inputs.cuda()
                    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                    outputs = model_G(inputs)
                    softmax_res = softmax(outputs.data.cpu().numpy()[0])
                    index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

                    inf_class = G_classes[index]
                elif inf_class == 'Mononuclear':
                    if M_transform is not None:
                        image = M_transform(org_image)
                    else:
                        image = org_image
                    inputs = image
                    with torch.no_grad():
                        inputs = Variable(inputs)

                    if use_gpu:
                        inputs = inputs.cuda()
                    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                    outputs = model_M(inputs)
                    softmax_res = softmax(outputs.data.cpu().numpy()[0])
                    index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

                    inf_class = M_classes[index]

                if (inf_class == inp):
                    cor += 1
                #writer.writerow({'file_name': file_path, 'prediction': inf_class}); tot += 1

print(cor/tot)
