#rm -rf /home/bumsoo/Data/resized/RBC/RBC/Crop_*

#python crop_RBC.py
rm -rf /home/bumsoo/Data/split/WBC_CAD/train/RBC/
rm -rf /home/bumsoo/Data/split/WBC_CAD/val/RBC/

mv /home/bumsoo/Data/split/RBC/train/RBC/ /home/bumsoo/Data/split/WBC_CAD/train/
mv /home/bumsoo/Data/split/RBC/val/RBC/ /home/bumsoo/Data/split/WBC_CAD/val/
rm -rf /home/bumsoo/Data/split/RBC
