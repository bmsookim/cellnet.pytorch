# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Module : 2_parser
# Description : XML_function.py
# The function code for XML file format handling.
# ***********************************************************

import os
import cv2
import sys
import numpy as np

from xml.etree.ElementTree import Element, SubElement, dump
from xml.etree.ElementTree import parse

keys = ["xmin", "ymin", "xmax", "ymax"]

# check whether the file is in XML data format
def is_XML(f):
    return f.endswith(".xml")

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

# read all the XML format files in the given directory
def read_all_XMLs(in_dir):
    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            file_path = os.path.join(subdir, f)
            if(is_XML(f)):
                print('{:<100}'.format(file_path))

# extract the bounding box list information from the XML data format
def parse_XML(xml):
    tree = parse(xml)
    root = tree.getroot()
    bbox_lst = []

    for obj in root.findall('object'):
        bndbox = obj.find("bndbox")
        cell_dict = {
            'xmin':0,
            'ymin':0,
            'xmax':0,
            'ymax':0,
        }

        for key in keys:
            cell_dict[key] = int(bndbox.find(key).text)

        bbox_lst.append(cell_dict)

    return bbox_lst

# outputs a image with all the bounding box representations drew within the image
def draw_cell_bbox(xml_dir, img_dir, out_base):
    for subdir, dirs, files in os.walk(xml_dir):
        for f in files:
            file_path = os.path.join(subdir, f)
            image_file = os.path.join(img_dir, f.split(".")[0] + ".png")
            background = cv2.imread(image_file)

            if(is_XML(f)):
                print('Processing {:<100}'.format(file_path))
                bbox_lst = parse_XML(file_path)
                for bbox in bbox_lst:
                    cv2.rectangle(background, (bbox["xmin"],bbox["ymin"]), (bbox["xmax"],bbox["ymax"]), (0,255,0), 1)

                cv2.imwrite(os.path.join(out_base, f.split(".")[0]+".png"), background)
