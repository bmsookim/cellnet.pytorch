import cv2
import csv
import os

from xml.etree.ElementTree import parse

base_dir = "/home/bumsoo/Data/BCCD/"
image_dir = os.path.join(base_dir, "JPEGImages")
annot_dir = os.path.join(base_dir, "Annotations")
split_dir = os.path.join(base_dir, "ImageSets", "Main")

train_txt = os.path.join(split_dir, "train.txt")
val_txt = os.path.join(split_dir, "val.txt")
test_txt = os.path.join(split_dir, "test.txt")

trainval_dir = "/home/bumsoo/Data/_train_val/BCCD/"

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

def read_text(in_file, has_indent=False):
    file_lst = []

    with open(in_file) as f:
        if has_indent:
            for line in f:
                file_lst.append(line[:-1])
        else:
            for line in f:
                file_lst.append(line)

    return file_lst

def parse_XML(xml):
    targetXML = open(xml, 'r')
    tree = parse(targetXML)
    root = tree.getroot()

    for element in root.findall('object'):
        name = element.find('name').text
        xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text),
            ['xmin', 'ymin', 'xmax', 'ymax']))

def convert_lst_to_dir(lst, mode):
    global image_dir, annot_dir

    if (mode == 'test'):
        base_dir = '/home/bumsoo/Data/_test/BCCD/'
    else:
        base_dir = '/home/bumsoo/Data/_train_val/BCCD/'

    for base in lst:
        img_dir = os.path.join(image_dir, base+".jpg")
        img = cv2.imread(img_dir)

        ann_dir = os.path.join(annot_dir, base+".xml")
        parse_XML(ann_dir)

def split_train_val(train_file, val_file, test_file, has_indent=False):
    train_lst = read_txt(train_file, has_indent)
    val_lst = read_txt(val_file, has_indent)
    test_lst = read_text(test_file, has_indent)

if __name__ == "__main__":
    with open('BCCD_labels.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            print(row)

    train_lst = read_text(train_txt, True)
    convert_lst_to_dir(train_lst, mode='train')
    check_and_mkdir('/home/bumsoo/Data/_train_val/BCCD')
    for d in ['train', 'val']:
        check_and_mkdir('/home/bumsoo/Data/_train_val/BCCD/%s' %d)

    check_and_mkdir('/home/bumsoo/Data/_test/BCCD')
