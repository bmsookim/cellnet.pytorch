import os
import cv2
import sys
import numpy as np

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

for file_number in range(1, 22):
    print("Predicting Bounding Box for TEST%d..." %file_number)
    original_img = cv2.imread('/home/bumsoo/Data/test/CT_20/TEST%d.png' %file_number)
    mask_img = cv2.imread('./masks/TEST%d.png' %file_number)
    check_and_mkdir("./cropped/TEST%d" %file_number)

    ret, threshed_img = cv2.threshold(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(threshed_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)

        if (area > 30**2):
            # ellipse
            #ellipse = cv2.fitEllipse(cnt)
            #cv2.ellipse(original_img, ellipse, (0,255,0), 2)

            x, y, w, h = cv2.boundingRect(cnt)

            print("\nAREA "+str(count))
            print(x,y)
            print(x+w, y+h)
            print(original_img.shape)

            #cv2.rectangle(original_img, (x, y), (x+w, y+h), (0,255,0), 2)
            """
            if (x > 20 and x+w < original_img.shape[1] and y > 20 and y+h < original_img.shape[0]):
                if (w < 80 and h < 80):
                    if (h > 1.2*w):
                        crop = original_img[y-20:y+h+20, x:x+w]
                    elif (w > 1.2*h):
                        crop = original_img[y:y+h, x-20:x+w+20]
                    else:
                        crop = original_img[y-20:y+h+20, x-20:x+w+20]
                else:
                    crop = original_img[y:y+h, x:x+w]
                count += 1
                cv2.imwrite("./cropped/TEST%d/%d.png" %(file_number, count), crop)
            else:
                print("WON")
                crop = original_img[y:y+h, x:x+w]
                count += 1
                cv2.imwrite("./cropped/TEST%d/%d.png" %(file_number, count), crop)
            """

            count += 1
            crop = original_img[y:y+h, x:x+w]
            cv2.imwrite("./cropped/TEST%d/%d.png" %(file_number, count), crop)
            cv2.rectangle(original_img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imwrite('./bbox/TEST%d.png' %file_number, original_img)
