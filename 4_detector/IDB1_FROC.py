import cv2
import os
import csv
import numpy as np

prefix = "/home/bumsoo/Data/ALL_IDB1"

TP, FP = 0, 0
total_pred = 0

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
    mask_img = cv2.imread('./results/ALL_IDB1/frcnn/' + f)
    original_img = img

    frcnn_IoU = []

    # IoU
    ground_truth_lst = []
    pred_lst = []

    # Ground Truth
    with open('%s/IDB1_annot/%s.csv' %(prefix, base), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            xmin, ymin, xmax, ymax = list(map(lambda x : int(x), row))

            cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            ground_truth_lst.append([xmin, ymin, xmax, ymax])

    # Prediction
    with open('./results/ALL_IDB1/frcnn/%s.csv' %base) as csvfile:
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

    return frcnn_IoU, original_img

def get_label_num(base):
    xyc = open('%s/xyc/%s%s' %(prefix, base, '.xyc'))
    lines = xyc.readlines()

    num_lines = len(lines) if len(lines) > 1 else 0

    return num_lines

def get_IoU_lst():
    pass

if __name__ == "__main__":
    total_TP = 0
    total_FP = 0
    threshold = 0.3

    for subdir, dirs, files in os.walk('%s/im' %prefix):
        for f in files:
            base = f.split(".")[0]
            print(base)

            P = get_label_num(base)
            IoU_lst, frcnn_img = frcnn_eval(os.path.join(subdir, f))
            total_TP += sum(1 for i in IoU_lst if i >= threshold)
            total_FP += sum(1 for i in IoU_lst if i < threshold)

            cv2.imwrite('./results/ALL_IDB1/frcnn_res/' + f, frcnn_img)

    print("Total TP = %d" %total_TP)
    print("Total FP = %d" %total_FP)
