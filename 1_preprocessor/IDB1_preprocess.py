import os
import cv2
import re
import numpy as np
import csv

prefix = "/home/bumsoo/Data/ALL_IDB1"

def is_image(f):
    return f.endswith(".jpg")

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

def convert_to_png(out_dir = prefix + '/png_im'):
    check_and_mkdir(out_dir)
    for subdir, dirs, files in os.walk('./im'):
        for f in files:
            print(f)
            img = cv2.imread(os.path.join(subdir, f))
            print(img.shape)


            save_name = os.path.join(out_dir, f.split(".")[0] + ".png")
            cv2.imwrite(save_name, img)

def mark_label(base, out_dir):
    print("reading %s/im/%s%s" %(prefix, base, '.jpg'))
    img = cv2.imread('%s/im/%s%s' %(prefix, base, '.jpg'))
    xyc = open('%s/xyc/%s%s' %(prefix, base, '.xyc'))
    lines = xyc.readlines()

    with open('./%s.csv' %base, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for line in lines:
            try:
                x, y, c = (re.split('\t|\n', line))
            except ValueError:
                continue

            xmin, ymin = list(map(lambda k : int(k)-50, [x,y]))
            xmax, ymax = list(map(lambda k : int(k)+50, [x,y]))
            print(xmin, ymin, xmax, ymax)
            writer.writerow([xmin, ymin, xmax, ymax])
            #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    #cv2.imwrite(os.path.join(out_dir, '%s.jpg' %(base)), img)
    xyc.close()

def color_label(base, out_dir):
    print("reading %s/im/%s%s" %(prefix, base, '.jpg'))
    img = cv2.imread('%s/im/%s%s' %(prefix, base, '.jpg'))
    blank = np.zeros((img.shape[0], img.shape[1]))
    xyc = open('%s/xyc/%s%s' %(prefix, base, '.xyc'))
    lines = xyc.readlines()
    for line in lines:
        try:
            x, y, c = (re.split('\t|\n', line))
        except ValueError:
            continue

        x, y = int(x), int(y)
        cv2.circle(blank, (x, y), 50, 255, -1)

    cv2.imwrite(os.path.join(out_dir, '%s.jpg' %(base)), blank)
    xyc.close()

def mark_labels(in_dir = '%s/im' %prefix):
    check_and_mkdir('%s/labeled/' %prefix)

    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            base = f.split(".")[0]
            print(base)

            mark_label(base, '%s/labeled/' %prefix)

def color_labels(in_dir = '%s/im' %prefix):
    check_and_mkdir('%s/colored/' %prefix)

    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            base = f.split(".")[0]
            print(base)

            color_label(base, '%s/colored/' %prefix)


if __name__ == "__main__":
    mark_labels()
    #color_labels()
