from PIL import Image

import argparse
import cv2
import os
import enter_label as el
import config as cf

refPt = []

# impath = el.select_image() # does not work currently
impath = cf.image_path
print("Uploading %s..." %(impath))
image = cv2.imread(impath)
clone = image.copy()
plot = image.copy()
cropping = False

window_name = cf.image_path.split("/")[-1]

def drag_and_drop(event, x, y, flags, param):
    global refPt, cropping, clone, plot

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        cv2.rectangle(plot, refPt[0], (x,y), (0,255,0), 2)
        cv2.imshow(window_name, plot)

        plot = image.copy()

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        cropping = False

        cv2.rectangle(plot, refPt[0], refPt[1], (0,255,0), 2)
        cv2.imshow(window_name, plot)

        has_quit = el.enter_label(impath, refPt)

        if(has_quit):
            plot = image.copy()
            cv2.imshow(window_name, plot)
        else:
            cv2.rectangle(image, refPt[0], refPt[1], (0,255,0), 2)

if __name__ == "__main__":
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, drag_and_drop)
    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            os.remove("result.txt")
            cv2.imshow(window_name, image)

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
