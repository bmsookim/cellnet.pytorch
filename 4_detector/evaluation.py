import argparse
import os
import cv2
import csv
import sys
import operator
import numpy as np
import config as cf

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image

count = 0
parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--start', default=1, type=int, help='starting index')
parser.add_argument('--finish', default=5, type=int, help='finishing index')
args = parser.parse_args()

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

def inference_crop(model, cropped_img):
    # check if cropped img is a torch Variable, if not, convert
    use_gpu = torch.cuda.is_available()

    if test_transform is not None:
        img = test_transform(Image.fromarray(cropped_img, mode='RGB'))

    inputs = img

    with torch.no_grad():
        inputs = Variable(inputs)

        if use_gpu:
            inputs = inputs.cuda()

        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

        outputs = model(inputs)
        softmax_res = softmax(outputs.data.cpu().numpy()[0])
        index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

    return index, score

def inference(original_img, mask_img, inference_csv, model):
    global count

    ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=4)

    _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fieldnames = ['prediction', 'x', 'y', 'w', 'h']
    writer = csv.DictWriter(inference_csv, fieldnames=fieldnames)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        x, y, w, h = cv2.boundingRect(cnt)
        crop = original_img[y:y+h, x:x+w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        idx, score = inference_crop(model, crop)

        answ = dset_classes[idx]
        #print("./%s_%s.png" %(answ, str(count)))
        #cv2.imwrite("./%s_%s.png" %(answ, str(count)), crop)
        count += 1

        writer.writerow({
            'prediction': answ,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })

def compute_IoU(img, back_img, pred_csv, answ_csv):
    """
    @ Input:
        csv files : ['prediction', 'x', 'y', 'w', 'h']
    """

    IoU = 0

    pred_reader = csv.reader(pred_csv)
    answ_reader = csv.reader(answ_csv)

    lst_A, lst_B = [], []

    for row in pred_reader:
        #print("Predictions")
        #print(row)
        lst_A.append(row)
        pred = row[0]

    for row in answ_reader:
        #print("Answers")
        #print(row)
        lst_B.append(row)
        label = row[0]

    has_printed_label, count_label, IoU_lst = False, 0, []
    for comp_A in lst_A:
        A_x, A_y, A_w, A_h = map(int, comp_A[1:])
        pred = comp_A[0]

        #print(pred)
        cv2.putText(back_img, "Pred = %s" %str(pred), (A_x+A_w, A_y+A_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.rectangle(back_img, (A_x, A_y), (A_x+A_w, A_y+A_h), (0,255,0), 2)

        for comp_B in lst_B:
            B_x, B_y, B_w, B_h = map(int, comp_B[1:])
            label = comp_B[0]

            in_x1 = max(A_x, B_x)
            in_x2 = min(A_x+A_w, B_x+B_w)
            in_w = in_x2 - in_x1

            in_y1 = max(A_y, B_y)
            in_y2 = min(A_y+A_h, B_y+B_h)
            in_h = in_y2 - in_y1

            if (in_w < 0 or in_h <0):
                interArea = 0
            else:
                interArea = in_w * in_h

            unionArea = (A_w*A_h) + (B_w*B_h) - interArea

            IoU = float(interArea) / float(unionArea)

            if (has_printed_label == False):
                count_label += 1
                cv2.putText(back_img, "Label = %s" %str(label), (B_x, B_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
                cv2.rectangle(back_img, (B_x, B_y), (B_x+B_w, B_y+B_h), (255, 0, 0), 2)
            if(IoU > 0):
                cv2.rectangle(back_img, (in_x1, in_y1), (in_x2, in_y2), (0,0,255), 2)
                cv2.putText(back_img, "IoU = %s" %str(IoU), (in_x1+int(in_w/2), in_y1+int(in_h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                IoU_lst.append(IoU)

        has_printed_label = True

    return float(sum(IoU_lst))/float(count_label), back_img

if __name__ == "__main__":
    for i in range(1, 3):
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

        in_dir = './results/%s/' %cf.name
        if not os.path.exists(in_dir):
            print("There is no result directory")
            sys.exit(1)

        img = cv2.imread('/home/bumsoo/Data/_test/Cell_Detect/TEST%d/TEST%d.png' %(i,i))
        mask_img = cv2.imread(in_dir + 'masks/TEST%d.png' %i)

        check_and_mkdir(in_dir + '/inferenced/')

        with open(in_dir + '/inferenced/TEST%d.csv' %i, 'w') as csvfile:
            inference(img, mask_img, csvfile, model)

        with open(in_dir + '/inferenced/TEST%d.csv' %i, 'r') as pred_csv:
            with open('/home/bumsoo/Data/_test/Cell_Detect/TEST%d/TEST%d.csv' %(i,i)) as answ_csv:
                back_img = img
                IoU, marked_img = compute_IoU(img, back_img, pred_csv, answ_csv)
                print("TEST#%d : Average IOU = %s" %(i, str(IoU)))
                cv2.imwrite(in_dir + "/inferenced/TEST%d.png" %i, marked_img)
