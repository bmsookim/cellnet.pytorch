from Tkinter import *
from PIL import Image
import tkFileDialog
import cv2

def select_image():
    global panelA, panelB

    path = tkFileDialog.askopenfilename()

if __name__ == "__main__":
    select_image()
