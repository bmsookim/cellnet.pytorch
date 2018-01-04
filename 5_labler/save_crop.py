import argparse
import cv2
import enter_label as el

refPt = []

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        cropping = False

        cv2.rectangle(image, refPt[0], refPt[1], (0,255,0), 2)
        cv2.imshow("image", image)

        el.enter_label("2.png")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):
        break

if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

cv2.destroyAllWindows()
