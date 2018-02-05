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
from time import sleep
from progressbar import *
from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image
from misc_functions import save_class_activation_on_image
from grad_cam import BackPropagation, GradCAM, GuidedBackPropagation

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--stepSize', default=50, type=int, help='size of each sliding window steps')
parser.add_argument('--windowSize', default=100, type=int, help='size of the sliding window')
parser.add_argument('--subtype', default='WBC_Neutrophil_Segmented', type=str, help='Type to find')
parser.add_argument('--testNumber', default=1, type=int, help='Test Number')
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

def random_crop(image, dim):
    if len(image.shape):
        W, H, D = image.shape
        w, h, d = dim
    else:
        W, H = image.shape
        w, h = size

    left, top = np.random.randint(W-w+1), np.random.randint(H-h+1)
    return image[left:left+w, top:top+h], left, top

def return_class_idx(class_name):
    global dset_classes

    for i,j in enumerate(dset_classes):
        if class_name == j:
            return i

    print(class_name + " is not an appropriate class to search.")
    sys.exit(1) # Wrong class name input

def generate_sliding_windows(image, stepSize, windowSize):
    list_windows = []

    for x in xrange(0, image.size[0], stepSize):
        for y in range(0, image.size[1], stepSize):
            if(x+windowSize < image.size[0] and y+windowSize < image.size[1]):
                list_windows.append(image.crop((x,y,x+windowSize,y+windowSize)))

    return list_windows

if __name__ == "__main__":
    # uploading the model
    print("| Loading checkpoint model for grad-CAM...")
    assert os.path.isdir('../3_classifier/checkpoint'),'[Error]: No checkpoint directory found!'
    assert os.path.isdir('../3_classifier/checkpoint/'+trainset_dir),'[Error]: There is no model weight to upload!'
    file_name = getNetwork(args)
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
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])

    #@ Code for extracting a grad-CAM region for a given class
    gcam = GradCAM(model._modules.items()[0][1], cuda=use_gpu)

    print(dset_classes)
    print("\n[Phase 2] : Gradient Detection")
    WBC_id = return_class_idx(args.subtype)

    print("| Checking Activated Regions for " + dset_classes[WBC_id] + "...")

    file_name = cf.test_dir + str(args.testNumber) + os.sep + ('TEST%s.png' %str(args.testNumber))
    #'./samples/full_image_%s.png' %str(args.testNumber)
    print("| Opening "+file_name+"...")

    original_image = cv2.imread(file_name)
    PIL_image = Image.open(file_name)

    lst = generate_sliding_windows(PIL_image, args.stepSize, args.windowSize)
    print("\n[Phase 3] : Sliding Window Heatmaps")
    heatmap_lst = []

    widgets = ['Heatmap Generated: ', Percentage(), ' ', Bar(marker='#', left='[', right=']'), ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(lst))
    pbar.start()
    progress = 0

    for img in lst:
        if (img.size[0] == img.size[1]): # Only consider foursquare regions
            backg = np.asarray(img)

            if test_transform is not None:
                img = test_transform(img)
                backg = cv2.resize(backg, (224, 224))

            inputs = img[:3,:,:]
            inputs = Variable(inputs, requires_grad=True)

            if use_gpu:
                inputs = inputs.cuda()
            inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

            #print(inputs.size())
            probs, idx = gcam.forward(inputs)

            # Grad-CAM
            gcam.backward(idx=WBC_id) # Get gradients for the selected label
            output = gcam.generate(target_layer='layer4.2') # Needs more testout

            item_id = (np.where(idx.cpu().numpy() == (WBC_id)))[0][0]
            if (probs[item_id] < 0.5):
                output = np.zeros_like(output)

            heatmap = cv2.cvtColor(np.uint8(output * 255.0), cv2.COLOR_GRAY2BGR)
            heatmap_lst.append(heatmap)
            #gcam.save("./heatmap/%s.png" %progress, heatmap, backg)
            #cv2.imwrite("./heatmap/%s.png" %progress, heatmap)
            #print('\t| {:5f}\t{}'.format(probs[0], dset_classes[idx[0]]))
            pbar.update(progress)
            progress += 1
    pbar.finish()

    print("\n[Phase 4] : Patching Up Individual Heatmaps")
    blank_canvas = np.zeros_like(original_image) # blank_canvas to draw the map on

    img_cnt = 0
    image = original_image

    for y in xrange(0, image.shape[0], args.stepSize):
        for x in range(0, image.shape[1], args.stepSize):
            f_map = heatmap_lst[img_cnt]
            f_map = cv2.resize(f_map, (args.windowSize, args.windowSize))
            target_window = blank_canvas[x:x+args.windowSize, y:y+args.windowSize]

            if (target_window.shape[0] == target_window.shape[1]): # Only for foursquare windows
                target_window += f_map
                img_cnt += 1

                if (img_cnt >= len(heatmap_lst)):
                    blank_canvas = cv2.medianBlur(blank_canvas, 9)
                    gcam.save('./results/%s_%s.png'
                            %(file_name.split(".")[-2].split("/")[-1], args.subtype), blank_canvas, original_image)
                    #cv2.imwrite('./results/original.png', original_image)
                    #gcam.save('./results/heat_final.png', blank_canvas, original_image)
                    print("| Feature map completed!")
                    sys.exit(0)
