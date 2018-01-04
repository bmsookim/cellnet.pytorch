import argparse
import cv2
import enter_label as el

refPt = []

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
clone = image.copy()
cropping = False

def drag_and_drop(event, x, y, flags, param):
    global refPt, cropping, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        cv2.rectangle(clone, refPt[0], (x,y), (0,255,0), 2)
        cv2.imshow("image", clone)

        clone = image.copy()

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        cropping = False

        cv2.rectangle(image, refPt[0], refPt[1], (0,255,0), 2)
        cv2.imshow("image", image)

        el.enter_label("2.png", refPt)

if __name__ == "__main__":
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", drag_and_drop)
    cv2.imshow("image", image)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
