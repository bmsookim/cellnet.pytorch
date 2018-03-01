#rm -rf /home/bumsoo/Data/resized/RBC/RBC/Crop_*

#python crop_RBC.py
rm -rf /home/bumsoo/Data/split/HJSB/train/RBC/
rm -rf /home/bumsoo/Data/split/HJSB/val/RBC/

mv /home/bumsoo/Data/split/RBC/train/RBC/ /home/bumsoo/Data/split/HJSB/train/
mv /home/bumsoo/Data/split/RBC/val/RBC/ /home/bumsoo/Data/split/HJSB/val/
rm -rf /home/bumsoo/Data/split/RBC
