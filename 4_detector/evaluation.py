import os
import cv2
import sys
import numpy as np
import config as cf

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating %s..." %in_dir)
        os.makedirs(in_dir)

def print_inference_with_labels(org, mask, labels):
    check_and_mkdir('./results/
