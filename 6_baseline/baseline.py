import cv2
import numpy as np
import os
import csv

def is_image(f):
    return f.endswith(".png") or f.endswith('.jpg') or f.endswith(".jpg")

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

file_list = os.listdir('/home/bumsoo/Data/_test/Guro/')

for f in file_list:
    img_dir = ('/home/bumsoo/Data/_test/Guro/%s/%s.png' %(f,f))
    print("Reading %s..." %img_dir)

    img = cv2.imread(img_dir)
    threshold_value = 190

    img_B = img[:,:,0]
    img_R = img[:,:,2]
    I_1 = img_R-img_B

    # threshold_value
    ret, thresh = cv2.threshold(I_1, threshold_value, 255, cv2.THRESH_BINARY)
    I_2 = np.invert(thresh)

    kernel_size_row = 3
    kernel_size_col = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row , kernel_size_col))

    erosion = cv2.erode(I_2,kernel,iterations=1)
    dilation = cv2.dilate(erosion,kernel,iterations=1)
    dilation = cv2.dilate(dilation,kernel,iterations=1)
    ret, dilation = cv2.threshold(dilation, 150, 255, cv2.THRESH_BINARY)
    I_3 = dilation==I_2
    _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    original_img = img.copy()

    area_array = []
    center_array = []
    xywh_array=[]

    # Initial bounding box
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 20**2:
            x,y,w,h = cv2.boundingRect(cnt)
            #cv2.rectangle(original_img, (x,y), (x+w,y+h), (0,255,0), 2)

            area_array.append(w*h)
            center_array.append([x+w/2,y+h/2])
            xywh_array.append([x,y,w,h])

    # max value of area : The union of the two area must be under max : get max area
    max_area = 0
    for area in area_array:
        if area > max_area:
            max_area = area

    del_list = []
    New_area = []
    New_center = []

    # For each area, merge
    for i in range(len(area_array)):
        for j in range(i+1,len(area_array)):
            area_1 = area_array[i]
            area_2 = area_array[j]

            cnt_1_center = center_array[i]
            cnt_2_center = center_array[j]

            distance = np.sqrt((cnt_1_center[0]-cnt_2_center[0])**2+(cnt_1_center[1]-cnt_2_center[1])**2)
            union_area = area_1+area_2

            # 1. close distance
            # 2. union area < max area
            if (distance < 70) and (union_area < max_area):
                New_area.append(union_area)
                New_center.append([(cnt_1_center[0]+cnt_2_center[0])/2,(cnt_1_center[1]+cnt_2_center[1])/2]) # add merged area
                del_list.append(i) # delete original area
                del_list.append(j) # delete original area

                print('distance %d - %d : %d'%(i,j,distance))
                print('union area %d - %d : %d'%(i,j,union_area))

    final_img = img.copy()

    check_and_mkdir('./results/%s' %str(threshold_value))
    csvfile = open(os.path.join('./results/',str(threshold_value), os.path.splitext(f)[0]+'.csv'), 'w', encoding='utf-8', newline='')
    wr = csv.writer(csvfile)

    # save initial area into csv file (merged area is already deleted from the list)
    for i in range(len(area_array)):
        if i not in del_list:
            x,y,w,h = xywh_array[i]

            end_x = x+w
            end_y = y+h

            if x<0:
                x=0
            if y<0:
                y=0
            if end_x>img.shape[1]:
                end_x = img.shape[1]
            if end_y>img.shape[0]:
                end_y = img.shape[0]

            wr.writerow(xywh_array[i]) # write (x,y,w,h) in csv file
            cv2.rectangle(final_img, (x,y), (end_x,end_y), (0,255,0), 2)

    # For each merged area
    for i in range(len(New_area)):
        area = New_area[i]
        center_x,center_y = New_center[i]

        w = np.sqrt(area)
        h = w

        if int(center_x-w/2) < 0:
            new_x = 0
            w = w + int(center_x-w/2)
        else:
            new_x = int(center_x-w/2)

        if int(center_y-h/2) < 0:
            new_y = 0
            h = h+int(center_y-h/2)
        else:
            new_y = int(center_y-h/2)

        if int(center_x+w/2) > img.shape[1]:
            new_end_x = img.shape[1]
            w = w - (int(center_x-w/2)-img.shape[1])
        else:
            new_end_x = int(center_x+w/2)

        if int(center_y+h/2) > img.shape[0]:
            new_end_y = img.shape[0]
            h = h - (int(center_y+h/2)-img.shape[0])
        else:
            new_end_y = int(center_y+h/2)

        wr.writerow([new_x,new_y,int(w),int(h)])
        cv2.rectangle(final_img, (new_x,new_y), (new_end_x,new_end_y), (0,255,0), 2)

    # close csv file
    csvfile.close()

    cv2.imwrite(os.path.join('./results/',str(threshold_value), os.path.splitext(f)[0]+"_"+'final_img.png'), final_img)
    print(str(threshold_value)+os.path.splitext(f)[0]+"_"+'final_img.png')

    dual_img = img.copy()

    for i in range(len(area_array)):
        x,y,w,h = xywh_array[i]
        cv2.rectangle(dual_img, (x,y), (x+w,y+h), (0,255,0), 2)


    for i in range(len(New_area)):
        area = New_area[i]
        center_x,center_y = New_center[i]

        w = np.sqrt(area)
        h = w

        cv2.rectangle(dual_img, (int(center_x-w/2),int(center_y-h/2)), (int(center_x+w/2),int(center_y+h/2)), (0,0,255), 2)

    cv2.imwrite(os.path.join('./results/',str(threshold_value), os.path.splitext(f)[0]+"_"+'dual_img.png'), dual_img)
    print(str(threshold_value)+os.path.splitext(f)[0]+"_"+'dual_img.png')
