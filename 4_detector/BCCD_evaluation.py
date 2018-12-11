import argparse
import os
import cv2
import csv
import sys
import operator
import numpy as np
import BCCD_config as cf

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image
from xml.etree.ElementTree import parse
annot_dir = os.path.join("/home/bumsoo/Data/BCCD", 'Annotations')

count = 0
parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
args = parser.parse_args()

frcnn_IoU = []
gcam_IoU = []

if (sys.version_info > (3,0)):
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])
else:
    test_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])

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

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating %s..." %in_dir)
        os.makedirs(in_dir)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def parse_XML(xml_dir):
    """
    @ Input : directory of XML file

    @ Function : Read XML format for a single whole image annotation, and
        1) Parse the tree and pass over the root
        2) Return a sorted list of the WBC in 'xmin' order for propoer label match for 'BCCD_labels.csv'
    """
    targetXML = open(xml_dir, 'r')
    tree = parse(targetXML)
    root = tree.getroot()
    WBC_lst = []

    for element in root.findall('object'):
        name = element.find('name').text
        if (name == 'WBC'):
            xmin = int(element.find('bndbox').find('xmin').text)
            WBC_lst.append(xmin)

    WBC_lst.sort()

    return root, WBC_lst

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB-xA+1)*max(0, yB-yA+1)

    box_A_area = (boxA[2] - boxA[0] + 1)*(boxA[3] - boxA[1] + 1)
    box_B_area = (boxB[2] - boxB[0] + 1)*(boxB[3] - boxB[1] + 1)
    iou = interArea / float(box_A_area + box_B_area - interArea)

    return iou

def frcnn_eval(file_path):
    base = f.split('.')[0]
    img = cv2.imread(file_path)
    mask_img = cv2.imread('./results/BCCD/masks/' + f)
    original_img = img

    # IoU
    ground_truth_lst = []
    pred_lst = []

    # Ground Truth
    xml_dir = os.path.join(annot_dir, base+".xml")
    root, WBC_lst = parse_XML(xml_dir)

    for element in root.findall('object'):
        name = element.find('name').text

        if name == 'WBC':
            xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text), ['xmin', 'ymin', 'xmax', 'ymax']))

            cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            ground_truth_lst.append([xmin, ymin, xmax, ymax])

    # Prediction
    with open('/home/bumsoo/Github/cellnet.pytorch/1_preprocessor/results/faster-rcnn/%s.csv' %base) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            name = row[0]

            if 'WBC' in name:
                x1, y1, x2, y2 = list(map(lambda x : int(x), row[1:]))

                cv2.rectangle(original_img, (x1,y1), (x2, y2), (0,255,0), 2)
                pred_lst.append([x1, y1, x2, y2])

    # If there exists a ground truth
    if len(ground_truth_lst) > 0:
        for gt in ground_truth_lst:
            if len(pred_lst) == 0:
                max_iou = 0
            else:
                iou_lst = list(map(lambda x : bb_intersection_over_union(x, gt), pred_lst))
                max_iou = max(iou_lst)
                del_idx = np.argmax(iou_lst)
                #del pred_lst[del_idx]

            frcnn_IoU.append(max_iou)
            cv2.putText(original_img, str(max_iou), (int((gt[0]+gt[2])/2), int((gt[1]+gt[3])/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        if len(pred_lst) > len(ground_truth_lst):
            for i in range(len(pred_lst)-len(ground_truth_lst)):
                frcnn_IoU.append(0)

    return original_img

def gcam_eval(file_path):
    base = f.split('.')[0]
    img = cv2.imread(file_path)
    mask_img = cv2.imread('./results/BCCD/masks/' + f)
    original_img = img

    # IoU calculation
    ground_truth_lst = []
    pred_lst = []

    # Ground Truth
    xml_dir = os.path.join(annot_dir, base+".xml")
    root, WBC_lst = parse_XML(xml_dir)

    for element in root.findall('object'):
        name = element.find('name').text

        if name == 'WBC':
            xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text), ['xmin', 'ymin', 'xmax', 'ymax']))

            cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            ground_truth_lst.append([xmin, ymin, xmax, ymax])

    # Prediction
    ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(original_img, (x,y), (x+w, y+h), (0,255,0), 2)
        pred_lst.append([x, y, x+w, y+h])

    # If there exists a ground truth
    if len(ground_truth_lst) > 0:
        for gt in ground_truth_lst:
            if len(pred_lst) == 0:
                max_iou = 0
            else:
                iou_lst = list(map(lambda x : bb_intersection_over_union(x, gt), pred_lst))
                max_iou = max(iou_lst)
                #del_idx = np.argmax(iou_lst)
                #del pred_lst[del_idx]

            gcam_IoU.append(max_iou)
            cv2.putText(original_img, str(max_iou), (int((gt[0]+gt[2])/2), int((gt[1]+gt[3])/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        if len(pred_lst) > len(ground_truth_lst):
            for i in range(len(pred_lst)-len(ground_truth_lst)):
                gcam_IoU.append(0)

    return original_img

if __name__ == "__main__":
    # Upload model
    trainset_dir = cf.data_base.split("/")[-1]+os.sep
    dsets = datasets.ImageFolder(os.path.join(cf.aug_base, 'train'))
    dset_classes = dsets.classes

    model_dir = '../3_classifier/checkpoint/'
    model_name = model_dir + trainset_dir
    file_name = getNetwork(args)

    assert os.path.isdir(model_dir), '[Error]: No checkpoint dir found!'
    assert os.path.isdir(model_name), '[Error]: There is no model weight to upload!'
    checkpoint = torch.load(model_name+file_name+".t7")
    model = checkpoint['model']

    check_and_mkdir('./results/BCCD/inferenced/')
    check_and_mkdir('./results/BCCD/frcnn')

    # Iterate files
    for subdir, dirs, files in os.walk(cf.test_base):
        for f in files:
            if f.endswith(".png") == False:
                continue

            in_dir = './results/BCCD/'
            if not os.path.exists(in_dir):
                print("There is no result directory")
                sys.exit(1)

            #img = cv2.imread(os.path.join(subdir, f))
            #mask_img = cv2.imread(in_dir + 'masks/' + f)
            #back_img = img

            #marked_img = bbox(back_img, mask_img)
            marked_img = gcam_eval(os.path.join(subdir, f))
            frcnn_img = frcnn_eval(os.path.join(subdir, f))
            print("Bounding Box Inference for %s" %f)
            cv2.imwrite(in_dir + "/inferenced/" + f, marked_img)
            cv2.imwrite(in_dir + "/frcnn/" + f, frcnn_img)

    avg_gcam_IoU = sum(gcam_IoU) / len(gcam_IoU)
    avg_frcnn_IoU = sum(frcnn_IoU) / len(frcnn_IoU)

    print(">>> Total <<<")
    print(len(gcam_IoU))
    print(len(frcnn_IoU))

    print(">>> Average <<<")
    print(avg_gcam_IoU)
    print(avg_frcnn_IoU)
