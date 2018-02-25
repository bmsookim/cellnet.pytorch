import cv2
import numpy as np
import os

def apply_INV_threshold(RGB_img, thresh_min=70, thresh_max=255):
    gray = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)

    return thresh

def drawBBOX(thresh, background):
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 10**2:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(background, (x,y), (x+w,y+h), (0,255,0), 2)

    return background

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

if __name__ == "__main__":
    in_dir = "/home/bumsoo/Data/test/CT_20/"
    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            if is_image(f):
                print("Applying Threshold to %s..." %f)
                file_path = os.path.join(subdir, f)
                img = cv2.imread(file_path)

                thresh = apply_INV_threshold(img)
                bbox_img = drawBBOX(thresh, img)

                cv2.imwrite("./results/%s" %f, bbox_img)
