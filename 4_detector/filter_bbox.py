import cv2
import os
import csv
import numpy as np

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

def return_thresh(img, thresh_min, thresh_max=255, is_gray=True):
    if(is_gray):
        input_img = img
    else:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(input_img, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)

    return thresh

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

in_dir = './results/cropped/'
out_dir = './results/filtered/'

check_and_mkdir(out_dir)

for file_number in range(1, 27+1):
    print("Filtering TEST%d..." %file_number)
    save_dir = os.path.join(out_dir, 'TEST%d' %file_number)
    check_and_mkdir(save_dir)
    with open(os.path.join(in_dir, 'TEST%d' %file_number, 'TEST%d.csv' %file_number), 'r') as csvfile:
        reader = csv.reader(csvfile)
        img = cv2.imread('/home/bumsoo/Data/test/CT_20/TEST%d.png' %file_number)
        count = 0
        for row in reader:
            count += 1
            x, y, w, h = map(int, row[1:])
            subtype = row[0]

            if ('Smudge' in subtype):
                pass
            else:
                pred = img[y:y+h, x:x+w]

                th_img = pred[:,:,2] - pred[:,:,0]

                thr = return_thresh(th_img, th_img.mean())
                _, contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                max_cnt = 0
                max_area = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if(area > max_area):
                        max_cnt = cnt
                        max_area = area

                save_path = os.path.join(save_dir, "%s_%d.png" %(subtype, count))

                if (max_area < ((pred.shape[0] * pred.shape[1])/3)):
                    pred_x, pred_y, pred_w, pred_h = cv2.boundingRect(max_cnt)
                    print("Image Cropped!!")
                    save_img = pred[pred_y:pred_y+pred_h, pred_x:pred_x+pred_h]

                else:
                    save_img = pred
