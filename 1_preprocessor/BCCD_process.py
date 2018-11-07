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
    """
    @ Input : in_dir

    @ Function : Check if 'in_dir' exists, if not create directory
    """
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

def parse_text(in_file, has_indent=False):
    """
    @ Input : .txt file

    @ Function : Read text format file and output a list of each rows
    """
    file_lst = []

    with open(in_file) as f:
        if has_indent:
            for line in f:
                file_lst.append(line[:-1])
        else:
            for line in f:
                file_lst.append(line)

    return file_lst

def parse_XML(xml_dir):
    """
    @ Input : directory of XML file

    @ Function : Read XML format for a single whole image annotation, and
        1) Parse the tree and pass over the root
        2) Return a sorted list of the WBC in 'xmin' order for propoer label match for 'BCCD_labels.csv'
    """
    targetXML = open(xml_dir, 'r')
    tree = parse(targetXML)
    root = tree.getroot()
    WBC_lst = []

    for element in root.findall('object'):
        name = element.find('name').text
        if (name == 'WBC'):
            xmin = int(element.find('bndbox').find('xmin').text)
            WBC_lst.append(xmin)

    WBC_lst.sort()

    return root, WBC_lst

def create_ground_truth(base, label):
    """
    @ Input : base, label_dict[base]

    @ Function :
        base -> img, xml
        Draw a bounding box represented ground truth for the given image & annotation
    """
    global image_dir, annot_dir

    # Image
    img_dir = os.path.join(image_dir, base+".jpg")
    img = cv2.imread(img_dir)

    # Annotation
    xml_dir = os.path.join(annot_dir, base+".xml")
    root, WBC_lst = parse_XML(xml_dir)

    for element in root.findall('object'):
        name = element.find('name').text
        xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text),
            ['xmin', 'ymin', 'xmax', 'ymax']))

        """
        if (name == 'RBC'):
            cls = name
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            cv2.putText(img, cls, (xmax, ymax),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
        """
        if (name == 'WBC'):
            cls = label
            if (',' in cls):
                cls = cls.split(',')[WBC_lst.index(xmin)]
            elif (len(WBC_lst) > 1):
                # Case 00113, More than 1 WBC even though no WBC label
                if (WBC_lst.index(xmin) > 0):
                    continue
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(img, cls, (xmin, int((ymin+ymax)/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

    check_and_mkdir("./BCCD_Ground_Truth/")
    #if len(WBC_lst) > 1 :
    cv2.imwrite("./BCCD_Ground_Truth/%s.png" %base, img)

def construct_lst_to_dir(lst, label_dict, mode):
    """
    @ Input : train_lst, val_lst, test_lst (list of base file names with no extensions like ".jpg")

    @ Function : Create
    """
    global image_dir, annot_dir

    if (mode == 'test'):
        base_dir = '/home/bumsoo/Data/_test/BCCD/'
    else:
        base_dir = '/home/bumsoo/Data/_train_val/BCCD/'

    for base in lst:
        create_ground_truth(base, label_dict[base])

def split_train_val(train_file, val_file, test_file, has_indent=False):
    train_lst = parse_text(train_file, has_indent)
    val_lst = parse_text(val_file, has_indent)
    test_lst = parse_text(test_file, has_indent)

    return train_lst, val_lst, test_lst

if __name__ == "__main__":
    label_dict = {}
    label_lst = []

    with open('BCCD_labels.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)

        for row in csv_reader:
            row[2] = row[2].replace(" ", "")
            label_dict['BloodImage_%05d' %(int(row[1]))] = (row[2])
            if("," in row[2]):
                for ind in row[2].split(','):
                    label_lst.append(ind)
            elif (row[2] != ''):
                label_lst.append(row[2])

    #print(label_dict)
    label_lst = set(label_lst)

    #train_lst = parse_text(train_txt, True)
    train_lst, val_lst, test_lst = split_train_val(train_txt, val_txt, test_txt, has_indent=True)

    # Construct directory from each lists.
    construct_lst_to_dir(train_lst, label_dict, mode='train')
    construct_lst_to_dir(val_lst, label_dict, mode='val')
    construct_lst_to_dir(test_lst, label_dict, mode='test')

    check_and_mkdir('/home/bumsoo/Data/_train_val/BCCD')
    for d in ['train', 'val']:
        #check_and_mkdir('/home/bumsoo/Data/_train_val/BCCD/%s' %d)
        for lbl in label_lst:
            check_and_mkdir('/home/bumsoo/Data/_train_val/BCCD/%s/%s' %(d,lbl))

    #check_and_mkdir('/home/bumsoo/Data/_test/BCCD')
    for lbl in label_lst:
        check_and_mkdir('/home/bumsoo/Data/_test/BCCD/%s' %lbl)
