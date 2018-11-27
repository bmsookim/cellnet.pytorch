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

            original_image = cv2.imread(file_name)
            PIL_image = Image.open(file_name)

            #lst = generate_sliding_windows(PIL_image, args.stepSize, args.windowSize)
            lst = [PIL_image.resize((224, 224), Image.ANTIALIAS)]

            print("\n[Phase 3] : Sliding Window Heatmaps")
            heatmap_lst = []

            widgets = ['Heatmap Generated: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='#', left='[', right=']'), ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(lst))
            pbar.start()
            progress = 0

            #csvname = 'logs/TEST%d.csv' %(args.testNumber) if args.subtype == None else 'logs/TEST%d_%s.csv' %(args.testNumber, args.subtype)

            #with open(csvname, 'w') as csvfile:
            fieldnames = ['location', 'prediction', 'score']
            #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
            for img in lst:
                if (img.size[0] == img.size[1]): # Only consider foursquare regions
                    backg = np.asarray(img)

                    if test_transform is not None:
                        img = test_transform(img)
                        backg = cv2.resize(backg, (in_size, in_size))

                    inputs = img[:3,:,:]
                    inputs = Variable(inputs, requires_grad=True)

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

                    if ('RBC' in dset_classes[idx[0]]  or probs[item_id] < 0.5):
                        heatmap_lst.append(np.uint8(np.zeros((in_size, in_size))))
                    else:
                        #print(dset_classes[comp_idx], probs[item_id].cpu().numpy())
                        """
                        writer.writerow({
                            'location': progress,
                            'prediction': dset_classes[comp_idx],
                            'score': probs[item_id][0]
                        })
                        """

                        # Grad-CAM
                        gcam.backward(idx=comp_idx) # Get gradients for the Top-1 label
                        output = gcam.generate(target_layer='layer4.2') # Needs more testout

                        #heatmap = cv2.cvtColor(np.uint8(output * 255.0), cv2.COLOR_GRAY2BGR)
                        heatmap = output
                        heatmap_lst.append(heatmap)
                    pbar.update(progress)
                    progress += 1
            pbar.finish()

            print("\n[Phase 4] : Patching Up Individual Heatmaps")

            img_cnt = 0
            image = original_image

            blank_canvas = np.zeros((image.shape[0], image.shape[1])) # blank_canvas to draw the mapo
            original_image = image
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            image = cv2.transpose(image)

            for x in xrange(0, image.shape[0], args.stepSize):
                for y in range(0, image.shape[1], args.stepSize):
                    print(len(heatmap_lst))
                    f_map = heatmap_lst[img_cnt]
                    f_map = cv2.resize(f_map, (args.windowSize, args.windowSize))

                    if (y+args.windowSize > image.shape[1] and x+args.windowSize > image.shape[0]):
                        target_window = blank_canvas[image.shape[1]-args.windowSize:image.shape[1], image.shape[0]-args.windowSize:image.shape[0]]
                    elif (y+args.windowSize > image.shape[1]):
                        target_window = blank_canvas[image.shape[1]-args.windowSize:image.shape[1], x:x+args.windowSize]
                    elif (x+args.windowSize > image.shape[0]):
                        target_window = blank_canvas[y:y+args.windowSize, image.shape[0]-args.windowSize:image.shape[0]]
                    else:
                        target_window = blank_canvas[y:y+args.windowSize, x:x+args.windowSize]

                    if (target_window.shape[0] == target_window.shape[1]): # Only for foursquare windows
                        target_window += f_map
                        img_cnt += 1

                        if (img_cnt >= len(heatmap_lst)):
                            check_and_mkdir('./results/%s' %cf.name)
                            check_and_mkdir('./results/%s/heatmaps/' %cf.name)
                            check_and_mkdir('./results/%s/masks/' %cf.name)
                            blank_canvas[blank_canvas > 1] = 1
                            blank_canvas = cv2.GaussianBlur(blank_canvas, (15,15), 0)
                            blank_save = np.uint8(blank_canvas * 255.0)

                            if args.subtype == None:
                                save_dir = './results/%s/heatmaps/%s.png' %(cf.name, file_name.split(".")[-2].split("/")[-1])
                                save_mask = './results/%s/masks/%s.png' %(cf.name, file_name.split(".")[-2].split("/")[-1])
                            else:
                                save_dir = './results/%s/heatmaps/%s_%s.png' %(cf.name, file_name.split(".")[-2].split("/")[-1], args.subtype)
                                save_mask = './results/%s/masks/%s_%s.png' %(cf.name, file_name.split(".")[-2].split("/")[-1], args.subtype)

                            # Save the grad-cam results
                            print("| Saving Heatmap results... ")
                            gcam.save(save_dir, blank_canvas, original_image) # save heatmaps
                            print("| Saving Mask results... ")
                            cv2.imwrite(save_mask, blank_save) # save image masks

                            print("| Feature map completed!\n")
                            # sys.exit(0)
