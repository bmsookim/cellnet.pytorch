import cv2
import os
import sys
import XML_function as Xf
import config as cf

def print_menu():
    print("\nSelect a mode by its name.\n")
    print("################## [ Options ] ###########################")
    print("# Mode 1 'print' : Print names of XML data file")
    print("# Mode 2 'bbox'  : Output the image with bounding box representation")
    print("# Mode 3 'exit'  : Terminate the program")
    print("##########################################################")


if __name__ == "__main__":
    while(1):
        print_menu()
        mode = raw_input('\nEnter mode name : ')

        ##############################################
        # @ Module 1 : Print names of image data file
        if (mode == 'print'):
            print(cf.XML_source)
            Xf.read_all_XMLs(cf.XML_source)

        #############################################
        # @ Module 2 : Retrieve Training data meanstd
        elif (mode == 'bbox'):
            Xf.check_and_mkdir(cf.result_source)
            Xf.draw_cell_bbox(cf.XML_source, cf.img_source, cf.result_source)

        #############################################
        elif (mode == 'exit'):
            print("\nGood Bye!\n")
            sys.exit(0)
        else:
            print("[Error] : Wrong input in 'mode name', please enter again.")
        #############################################
