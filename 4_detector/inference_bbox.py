import os
import cv2
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import argparse
import config as cf
import operator

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--start', default=1, type=int, help='starting index')
parser.add_argument('--finish', default=21, type=int, help='finishing index')
args = parser.parse_args()

# Phase 1 : Model Upload
print('\n[Test Phase] : Model Weight Upload')
use_gpu = torch.cuda.is_available()

# upload labels
data_dir = cf.aug_base
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
print("| Loading checkpoint model for crop inference...")
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

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean, cf.std)
])

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

for file_number in range(args.start, args.finish+1):
    print("| Predicting Box Inference for TEST%d..." %file_number)
    original_img = cv2.imread('/home/bumsoo/Data/test/CT_20/TEST%d.png' %file_number)
    mask_img = cv2.imread('./results/masks/TEST%d.png' %file_number)

    check_and_mkdir("./results/cropped/")
    check_and_mkdir("./results/cropped/TEST%d" %file_number)

    ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=4)

    _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)

        if (area > 30**2):
            x, y, w, h = cv2.boundingRect(cnt)
            crop = original_img[y:y+h, x:x+w]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # Swap the image into RGB input

            if test_transform is not None:
                img = test_transform(Image.fromarray(crop, mode='RGB'))

            inputs = img
            inputs = Variable(inputs, volatile=True)

            if use_gpu:
                inputs = inputs.cuda()
            inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

            outputs = model(inputs)
            softmax_res = softmax(outputs.data.cpu().numpy()[0])
            index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

            count += 1
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            if ('RBC' in dset_classes[index]):
                print("\tRBC_%d : %f" %(count, score))
            else:
                cv2.imwrite("./results/cropped/TEST%d/%s_%d.png" %(file_number, dset_classes[index], count), crop)
                print("\t%s_%d : %f" %(dset_classes[index], count, score))
