import cv2
import csv
import os

from xml.etree.ElementTree import parse
from BCCD_process import check_and_mkdir, parse_text, parse_XML, create_ground_truth
from shutil import copyfile

base_dir = "/home/bumsoo/Data/BCCD/"
image_dir = os.path.join(base_dir, "JPEGImages")
annot_dir = os.path.join(base_dir, "Annotations")
split_dir = os.path.join(base_dir, "ImageSets", "Main")
infer_dir = "./results/faster-rcnn"
test_dir = "/home/bumsoo/Data/_test/BCCD_FULL"

test_txt = os.path.join(split_dir, "test.txt")

def sort_test_results(test_txt, in_dir, out_dir):
    """

    """
    test_lst = parse_text(test_txt, has_indent=True)

    dst = out_dir
    check_and_mkdir(dst)

    for item in test_lst:
        print(item)
        src_dir = os.path.join(in_dir, item+".csv")
        dst_dir = os.path.join(dst, item+".csv")

        copyfile(src_dir, dst_dir)

    return test_lst

def faster_RCNN_eval(csvdir, annot_dir):
    print(sort_test_results(test_txt=test_txt, in_dir=csvdir, out_dir=infer_dir))

def draw_WBC_inference_and_annot(img_dir=image_dir, inf_dir=infer_dir, ann_dir=annot_dir, out_dir=infer_dir):
    """
    @ Input :
        img_dir - image directory
        inf_dir - inference csv directory
        ann_dir - annotation xml directory
    """

    test_lst = parse_text(test_txt, has_indent=True)
    check_and_mkdir(test_dir)

    for item in test_lst:
        print(os.path.join(img_dir, item+'.jpg'))
        img = cv2.imread(os.path.join(img_dir, item+".jpg"))
        cv2.imwrite(os.path.join(test_dir, item+'.png'), img)
        xml_dir = os.path.join(annot_dir, item+".xml")
        root, WBC_lst = parse_XML(xml_dir)

        # mark annotations
        for element in root.findall('object'):
            name = element.find('name').text
            xmin, ymin, xmax, ymax = list(map(lambda x: int(element.find('bndbox').find(x).text),
                ['xmin', 'ymin', 'xmax', 'ymax']))

            if name == 'WBC':
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(img, "ground truth", (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # mark inference
        with open(os.path.join(inf_dir, item+'.csv'), 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                if 'WBC' in row[0]:
                    xmin, ymin, xmax, ymax = list(map(lambda x: int(x), row[1:]))

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                    cv2.putText(img, "prediction", (xmin, ymin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, item+"-result.png"), img)

if __name__ == "__main__":
    csv_dir = "/home/bumsoo/Github/keras-frcnn/results_csv"
    #faster_RCNN_eval(csv_dir, "./")
    draw_WBC_inference_and_annot()
