# coding: utf-8


import csv
import sys
import cv2
from os import makedirs
from os.path import splitext,basename,exists
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        
        self.ui = uic.loadUi("Cell_labler.ui", self)
        self.ui.show()

        self.Init_open = True
        self.mouse_x = 0
        self.mouse_y = 0
        self.setMouseTracking(True)
        self.ui.label.setMouseTracking(True)

        self.ui.pushButton_2.setShortcut("Ctrl+a")
        self.ui.pushButton_3.setShortcut("Ctrl+z")
        self.ui.pushButton_4.setShortcut("Ctrl+s")

        self.current_table = 0
        self.box_start=False
        self.BOOLfile_open = False

        self.ui.radioButton.toggled.connect(lambda:self.btnstate(self.ui.radioButton))
        self.ui.radioButton_2.toggled.connect(lambda:self.btnstate(self.ui.radioButton_2))
        self.ui.radioButton_3.toggled.connect(lambda:self.btnstate(self.ui.radioButton_3))
        self.ui.radioButton_4.toggled.connect(lambda:self.btnstate(self.ui.radioButton_4))
        self.ui.radioButton_5.toggled.connect(lambda:self.btnstate(self.ui.radioButton_5))
        self.ui.radioButton_6.toggled.connect(lambda:self.btnstate(self.ui.radioButton_6))
        self.ui.radioButton_7.toggled.connect(lambda:self.btnstate(self.ui.radioButton_7))
        self.ui.radioButton_8.toggled.connect(lambda:self.btnstate(self.ui.radioButton_8))
        self.ui.radioButton_9.toggled.connect(lambda:self.btnstate(self.ui.radioButton_9))
        self.ui.radioButton_10.toggled.connect(lambda:self.btnstate(self.ui.radioButton_10))
        self.ui.radioButton_11.toggled.connect(lambda:self.btnstate(self.ui.radioButton_11))
        self.ui.radioButton_12.toggled.connect(lambda:self.btnstate(self.ui.radioButton_12))
        self.ui.radioButton_13.toggled.connect(lambda:self.btnstate(self.ui.radioButton_13))
        self.ui.radioButton_14.toggled.connect(lambda:self.btnstate(self.ui.radioButton_14))
        self.ui.radioButton_15.toggled.connect(lambda:self.btnstate(self.ui.radioButton_15))
        self.ui.radioButton_16.toggled.connect(lambda:self.btnstate(self.ui.radioButton_16))
        self.ui.radioButton_17.toggled.connect(lambda:self.btnstate(self.ui.radioButton_17))
        self.ui.radioButton_18.toggled.connect(lambda:self.btnstate(self.ui.radioButton_18))



    def save_table(self):
        directory_name = splitext(self.file_name)[0]
        
        pixmap_image_for_save = QtGui.QPixmap(self.file_name)

        if not exists(directory_name):
            makedirs(directory_name)

        csv_name = splitext(self.file_name)[0]+'.csv'
        f = open(csv_name,'w',encoding='utf-8',newline='')
        wr = csv.writer(f)
        for i in range(self.current_table):
            data_row=[]
            for j in range(self.ui.info_table.columnCount()):
                data_row.append(self.ui.info_table.item(i,j).text())
            crop_box = QtCore.QRect(int(data_row[1]),int(data_row[2]),int(data_row[3]),int(data_row[4]))
            crop_image = pixmap_image_for_save.copy(crop_box)
            crop_image.save(directory_name+'/'+basename(splitext(self.file_name)[0])+'_'+str(i)+'_'+data_row[0]+'.png','png')
            wr.writerow(data_row)
        f.close()
        self.ui.Announcement.setText('Saved at %s'%csv_name)


    def btnstate(self,radioBTN):
        if radioBTN.isChecked():
            self.ui.cell_type.setText('%s'%radioBTN.text())

    def add_table(self):
        self.ui.info_table.setRowCount(self.current_table + 1)

        
        self.ui.Announcement.setText("")
        self.ui.info_table.setItem(self.current_table,0,
                    QtWidgets.QTableWidgetItem(self.cell_type.text()))
        self.ui.info_table.setItem(self.current_table,1,
                    QtWidgets.QTableWidgetItem(str(self.box_X)))
        self.ui.info_table.setItem(self.current_table,2,
                    QtWidgets.QTableWidgetItem(str(self.box_Y)))
        self.ui.info_table.setItem(self.current_table,3,
                    QtWidgets.QTableWidgetItem(str(self.box_width)))
        self.ui.info_table.setItem(self.current_table,4,
                    QtWidgets.QTableWidgetItem(str(self.box_height)))
        self.current_table += 1

    def delete_table(self):
        self.current_table -= 1

        self.ui.Announcement.setText("")
 
        self.ui.info_table.setItem(self.current_table,0,
                    QtWidgets.QTableWidgetItem(str()))
        
        self.ui.info_table.setItem(self.current_table,1,
                    QtWidgets.QTableWidgetItem(str()))
        self.ui.info_table.setItem(self.current_table,2,
                    QtWidgets.QTableWidgetItem(str()))
        self.ui.info_table.setItem(self.current_table,3,
                    QtWidgets.QTableWidgetItem(str()))
        self.ui.info_table.setItem(self.current_table,4,
                    QtWidgets.QTableWidgetItem(str()))
       
        self.ui.info_table.setRowCount(self.current_table)


    def mouseMoveEvent(self, QMouseEvent):
            
        if self.BOOLfile_open and self.box_start:
            del self.painterInstance_tmp
            self.pixmap_image_tmp = QtGui.QPixmap(self.file_name)
            self.painterInstance_tmp = QtGui.QPainter(self.pixmap_image_tmp)

            self.Finish_x = QMouseEvent.x()
            self.Finish_y = QMouseEvent.y()
            
            if not ((self.Start_x == self.Finish_x) or
                (self.Start_y == self.Finish_y) or
                ((min(self.Start_x,self.Finish_x)-self.ui.label.geometry().x())<0) or
                ((min(self.Start_y,self.Finish_y)-self.ui.label.geometry().y())<0)):

                
                self.box_X = min(self.Start_x,self.Finish_x)-self.ui.label.geometry().x()
                self.box_X = self.box_X*self.width_scale_ratio
                self.box_Y = min(self.Start_y,self.Finish_y)-self.ui.label.geometry().y()
                self.box_Y = self.box_Y*self.height_scale_ratio
                self.box_width = max(self.Start_x,self.Finish_x)-min(self.Start_x,self.Finish_x)
                self.box_width = self.box_width*self.width_scale_ratio
                self.box_height = max(self.Start_y,self.Finish_y)-min(self.Start_y,self.Finish_y) 
                self.box_height = self.box_height*self.height_scale_ratio

                self.box_X = int(float(self.box_X))
                self.box_Y = int(float(self.box_Y))
                self.box_width = int(float(self.box_width))
                self.box_height = int(float(self.box_height))

                if not ((self.Start_x==self.Finish_x) or
                    (self.Start_y==self.Finish_y) or
                    ((self.box_X+self.box_width)>self.pixmap_image.width()) or
                    ((self.box_Y+self.box_height)>self.pixmap_image.height()) or
                    (self.box_X<0)or(self.box_Y<0)):
                 
                    self.Finish_pos.setText("""
                        X      : %s
                        Y      : %s
                        Width  : %s
                        Height : %s
                        """%(str(self.box_X),
                        str(self.box_Y),
                        str(self.box_width),
                        str(self.box_height)
                        ))
                    
                    self.penRectangle_tmp = QtGui.QPen(QtCore.Qt.red)
                    self.penRectangle_tmp.setWidth(1)

                    self.painterInstance_tmp.setPen(self.penRectangle_tmp)
                    
                    self.box_rect_tmp = QtCore.QRect(self.box_X,self.box_Y,self.box_width,self.box_height)
                    self.painterInstance_tmp.drawRect(self.box_rect_tmp)

                    self.ui.label.setPixmap(self.pixmap_image_tmp)
                    self.ui.label.show()
            
         


    def mousePressEvent(self,QMouseEvent):
        self.Start_x = QMouseEvent.x()
        self.Start_y = QMouseEvent.y()
        self.box_start = True

    def mouseReleaseEvent(self,QMouseEvent):
        self.box_start = False
        self.Finish_x = QMouseEvent.x()
        self.Finish_y = QMouseEvent.y()
       
        if not ((self.Start_x == self.Finish_x) or
                (self.Start_y == self.Finish_y) or
                ((min(self.Start_x,self.Finish_x)-self.ui.label.geometry().x())<0) or
                ((min(self.Start_y,self.Finish_y)-self.ui.label.geometry().y())<0)):


            self.box_X = min(self.Start_x,self.Finish_x)-self.ui.label.geometry().x()
            self.box_X = self.box_X*self.width_scale_ratio
            self.box_Y = min(self.Start_y,self.Finish_y)-self.ui.label.geometry().y()
            self.box_Y = self.box_Y*self.height_scale_ratio
            self.box_width = max(self.Start_x,self.Finish_x)-min(self.Start_x,self.Finish_x)
            self.box_width = self.box_width*self.width_scale_ratio
            self.box_height = max(self.Start_y,self.Finish_y)-min(self.Start_y,self.Finish_y) 
            self.box_height = self.box_height*self.height_scale_ratio

            self.box_X = int(float(self.box_X))
            self.box_Y = int(float(self.box_Y))
            self.box_width = int(float(self.box_width))
            self.box_height = int(float(self.box_height))


            if not ((self.Start_x==self.Finish_x) or
                    (self.Start_y==self.Finish_y) or
                    ((self.box_X+self.box_width)>self.pixmap_image.width()) or
                    ((self.box_Y+self.box_height)>self.pixmap_image.height()) or
                    (self.box_X<0)or(self.box_Y<0)):
                self.Finish_pos.setText("""
                    X      : %s
                    Y      : %s
                    Width  : %s
                    Height : %s
                    """%(str(self.box_X),
                    str(self.box_Y),
                    str(self.box_width),
                    str(self.box_height)
                    ))

                self.penRectangle = QtGui.QPen(QtCore.Qt.red)
                self.penRectangle.setWidth(0.1)

                self.painterInstance.setPen(self.penRectangle)
                self.box_rect = QtCore.QRect(self.box_X,self.box_Y,self.box_width,self.box_height)
                self.painterInstance.drawRect(self.box_rect)

                self.ui.label.setPixmap(self.pixmap_image)
                self.ui.label.show()






    @pyqtSlot()
    def open_image(self):
        if self.Init_open:
            self.file_name,_ = QFileDialog.getOpenFileName(self,'Open File', '.')
            
            if splitext(self.file_name)[-1] not in ['.png']:
                img = cv2.imread(self.file_name)
                self.file_name = splitext(self.file_name)[0] + '.png'
                cv2.imwrite(self.file_name,img)

            self.setWindowTitle(self.file_name)
            self.pixmap_image = QtGui.QPixmap(self.file_name)
            self.pixmap_image_tmp = QtGui.QPixmap(self.file_name)

            self.width_scale_ratio = self.pixmap_image.width()/self.ui.label.geometry().width()
            self.height_scale_ratio = self.pixmap_image.height()/self.ui.label.geometry().height()

            self.painterInstance = QtGui.QPainter(self.pixmap_image)
            self.painterInstance_tmp = QtGui.QPainter(self.pixmap_image_tmp)
            
            self.ui.label.setPixmap(self.pixmap_image)
            self.ui.label.show()
            self.Init_open = False
            
            self.BOOLfile_open = True 

        else:
            del self.painterInstance
            del self.pixmap_image
            
            self.file_name,_ = QFileDialog.getOpenFileName(self,'Open File', '.')
            self.setWindowTitle(self.file_name)
            self.pixmap_image = QtGui.QPixmap(self.file_name)
            
            self.width_scale_ratio = self.pixmap_image.width()/self.ui.label.geometry().width()
            self.height_scale_ratio = self.pixmap_image.height()/self.ui.label.geometry().height()
            
            self.painterInstance = QtGui.QPainter(self.pixmap_image)
            self.painterInstance_tmp = QtGui.QPainter(self.pixmap_image)
            

            self.ui.label.setPixmap(self.pixmap_image)
            self.ui.label.show()
            self.Init_open = False
            self.BOOLfile_open = True
            

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    w.show()
    sys.exit(app.exec())

