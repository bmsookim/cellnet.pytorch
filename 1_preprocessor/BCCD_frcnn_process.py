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

def save_and_return(base, phase, csvfile):
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

    fieldnames = ['filepath', 'x1', 'y1', 'x2', 'y2', 'class_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for element in root.findall('object'):
        name = element.find('name').text
        xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text),
            ['xmin', 'ymin', 'xmax', 'ymax']))

        if (name == 'WBC'):
            writer.writerow({
                'filepath': "./train_images/%s.jpg" %base,
                'x1': xmin,
                'y1': ymin,
                'x2': xmax,
                'y2': ymax,
                'class_name': name
            })

    cv2.imwrite("./%s_images/%s.jpg" %(phase, base), img)

def split_train_val(train_file, val_file, test_file, has_indent=False):
    train_lst = parse_text(train_file, has_indent)
    val_lst = parse_text(val_file, has_indent)
    test_lst = parse_text(test_file, has_indent)

    return train_lst, val_lst, test_lst

def create_csv_to_txt(in_csv, out_txt):
    csv_file = in_csv
    txt_file = out_txt

    text_list = []

    with open(csv_file, "r") as my_input_file:
        for line in my_input_file:
            line = line.split(",")
            text_list.append(",".join(line))

    with open(txt_file, "w") as my_output_file:
        for line in text_list:
            my_output_file.write(line)
    print('File Successfully written.')

if __name__ == "__main__":
    label_dict = {}
    label_lst = []

    with open('BCCD_labels.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)

        count = 0
        for row in csv_reader:
            count += 1
            #print('%s : %s' %(str(count), row[2]))
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
    for phase in ['train']:
        check_and_mkdir('./%s_images' %phase)
        with open('%s.csv' %phase, 'w') as csvfile:
            for base in train_lst:
                save_and_return(base, phase, csvfile)

    create_csv_to_txt('train.csv', 'annotate.txt')
