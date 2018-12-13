import cv2
import csv
import os
import random
import numpy as np

from xml.etree.ElementTree import parse
from BCCD_frcnn_process import base_dir, image_dir, annot_dir, split_dir, train_txt, val_txt, test_txt
from BCCD_frcnn_process import check_and_mkdir, parse_text, parse_XML, split_train_val, create_csv_to_txt

patch_num_lst = [3, 4]

def lst_to_img(patch_num, patch_lst, path_img_lst, w, h, f_base, csvfile):
    blank_canvas = np.zeros((w*patch_num, h*patch_num, 3))
    mark_canvas = np.zeros((w*patch_num, h*patch_num, 3))
    print(blank_canvas.shape)

    cnt = 0
    file_path = os.path.join('./augmented', f_base+"-%s.jpg" %patch_num)
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(patch_num):
        for j in range(patch_num):
            starting_point = (w*i, j*h)
            blank_canvas[starting_point[0]:starting_point[0]+w, starting_point[1]:starting_point[1]+h] = patch_img_lst[cnt]
            mark_canvas[starting_point[0]:starting_point[0]+w, starting_point[1]:starting_point[1]+h] = patch_img_lst[cnt]
            base = patch_lst[cnt]
            xml_dir = os.path.join(annot_dir, base+".xml")
            root, WBC_lst = parse_XML(xml_dir)

            for element in root.findall('object'):
                name = element.find('name').text
                if (name == 'WBC'):
                    xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text), ['xmin', 'ymin', 'xmax', 'ymax']))
                    xmin, xmax = list(map(lambda x: x+starting_point[1], [xmin, xmax]))
                    ymin, ymax = list(map(lambda x: x+starting_point[0], [ymin, ymax]))
                    writer.writerow([file_path, xmin, ymin, xmax, ymax, 'WBC'])
                    cv2.rectangle(mark_canvas, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cnt += 1

    cv2.imwrite('./test/%s-%d.jpg' %(f_base, patch_num), blank_canvas)
    cv2.imwrite('./test_gt/%s-%d.jpg' %(f_base, patch_num), mark_canvas)

def check_lst(test_lst):
    for base in test_lst:
        xml_dir = os.path.join(annot_dir, base+".xml")
        root, WBC_lst = parse_XML(xml_dir)

        cnt = 0
        for element in root.findall('object'):
            name = element.find('name').text
            if (name == 'WBC'):
                cnt += 1

        if cnt == 0:
            test_lst.remove(base)

    return test_lst

# train 205 samples
if __name__ == "__main__":
    label_dict = {}
    label_lst = []
    trainval_txt = os.path.join(split_dir, "trainval.txt")

    train_lst, val_lst, test_lst = split_train_val(trainval_txt, val_txt, test_txt, has_indent=True)

    # Generate 3000 samples
    sample = cv2.imread(os.path.join(image_dir, train_lst[0]+".jpg"))
    print(sample.shape)

    w = sample.shape[0]
    h = sample.shape[1]

    check_and_mkdir('./test/')
    check_and_mkdir('./test_gt')

    with open('test.csv', 'w') as csvfile:
        test_lst = check_lst(test_lst)
        print(len(test_lst))
        for first_base in test_lst:
            for patch_num in patch_num_lst:
                patch_lst = []
                patch_lst.append(first_base)
                #tmp = test_lst #call by reference
                tmp = test_lst[:]
                tmp.remove(first_base)
                sample_lst = random.sample(tmp, patch_num**2-1)

                patch_lst += sample_lst
                patch_img_lst = list(map(lambda x : cv2.imread(os.path.join(image_dir, x+".jpg")), patch_lst))
                print(len(patch_img_lst))
                lst_to_img(patch_num, patch_lst, patch_img_lst, w, h, first_base, csvfile)

    create_csv_to_txt('test.csv', 'annotate.txt')
