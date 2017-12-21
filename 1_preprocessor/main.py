import cv2
import os
import sys
import file_function as ff
import config as cf

def print_menu():
    print("\nSelect a mode by its name.\n")
    print("################## [ Options ] ###########################")
    print("# Mode 1 'print' : Print names of image data file")
    print("# Mode 2 'read'  : [original/aug] Read names data")
    print("# Mode 3 'resize': [target_size]  Resize & Orgnaize data")
    print("# Mode 4 'split' : Create a train-validation split of data")
    print("# Mode 5 'count' : Check the distribution of raw data")
    print("# Mode 6 'check' : Check the distribution of train/val split")
    print("# Mode 6 'exit'  : Terminate the program")
    print("##########################################################")


if __name__ == "__main__":
    while(1):
        print_menu()
        mode = raw_input('\nEnter mode name : ')

        ##############################################
        # @ Module 1 : Print names of image data file
        if (mode == 'print'):
            ff.print_all_imgs(cf.data_base)

        #############################################
        # @ Module 2 : Read all images
        elif (mode == 'read'):
            path = raw_input('Enter [original/resized] : ')
            if (not path in ['original', 'resized']):
                print("[Error] : Please define the mode between [original/resized].")
            else:
                if(path == 'original'):
                    ff.read_all_imgs(cf.data_base)
                elif(path == 'resized'):
                    ff.read_all_imgs(cf.resize_dir)

        #############################################
        # @ Module 3 : Resize and check images
        elif (mode == 'resize'):
            ff.check_and_mkdir(cf.resize_base)
            target_size = int(raw_input('Enter size : '))
            ff.resize_images(cf.data_base, cf.resize_dir, target_size)
            # ff.resize_and_contrast(cf.data_base, cf.resize_dir, target_size)

        elif (mode == 'test'):
            if (len(sys.argv) < 3):
                print("[Error] : Please define size in the second argument.")
            elif (len(sys.argv) < 4):
                print("[Error] : Please define data folder name in the third argument.")
            else:
                cf.data_base = "/mnt/datasets/" + sys.argv[3]
                cf.resize_dir = "/home/bumsoo/Data/test/" + sys.argv[3]
                target_size = int(sys.argv[2])
                ff.resize_images(cf.data_base, cf.resize_dir, target_size)

        #############################################
        # @ Module 4 : Train-Validation split
        elif (mode == 'split'):
            ff.check_and_mkdir(cf.split_base)
            split_dir = ff.create_train_val_split(cf.resize_dir, cf.split_dir)
            print("Train-Validation split directory = " + cf.split_dir)

        ############################################
        # @ Module 5 : Check the dataset
        elif (mode == 'count'):
            print("| " + cf.resize_dir.split("/")[-1] + " dataset : ")
            ff.count_each_class(cf.resize_dir)
        elif (mode == 'check'):
            ff.get_split_info(cf.split_dir)

        ############################################
        # @ Module 6 : Training data augmentation
        elif (mode == 'aug'):
            if (len(sys.argv) < 3):
                print("[Error] : Please define size in the second arguement.")
            else:
                ff.aug_train(cf.split_dir, sys.argv[2])

        #############################################
        # @ Module 7 : Retrieve Training data meanstd
        elif (mode == 'meanstd'):
            mean = ff.train_mean(cf.split_dir)
            print(mean)
            std = ff.train_std(cf.split_dir, mean)
            print(std)

        #############################################
        elif (mode == 'exit'):
            print("\nGood Bye!\n")
            sys.exit(0)
        else:
            print("[Error] : Wrong input in 'mode name', please enter again.")
        #############################################
