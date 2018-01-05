# coding: utf-8


import csv
import sys
from os.path import splitext
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
      
        self.current_table = 0
        self.box_start=False

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
        csv_name = splitext(self.file_name)[0]+'.csv'
        f = open(csv_name,'w',encoding='utf-8',newline='')
        wr = csv.writer(f)
        for i in range(self.current_table):
            data_row=[]
            for j in range(self.ui.info_table.columnCount()):
                data_row.append(self.ui.info_table.item(i,j).text())
            wr.writerow(data_row)
        f.close()
        self.ui.Announcement.setText('Saved at %s'%csv_name)


    def btnstate(self,radioBTN):
        if radioBTN.isChecked():
            self.ui.cell_type.setText('%s'%radioBTN.text())

    def add_table(self):
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
       


    def mouseMoveEvent(self, QMouseEvent):
        '''
        if self.box_start:

            self.Finish_x = QMouseEvent.x()
            self.Finish_y = QMouseEvent.y()
            
            self.box_X = min(self.Start_x,self.Finish_x)-self.ui.label.geometry().x()
            self.box_X = self.box_X*self.width_scale_ratio
            self.box_Y = min(self.Start_y,self.Finish_y)-self.ui.label.geometry().y()
            self.box_Y = self.box_Y*self.height_scale_ratio
            self.box_width = max(self.Start_x,self.Finish_x)-min(self.Start_x,self.Finish_x)
            self.box_width = self.box_width*self.width_scale_ratio
            self.box_height = max(self.Start_y,self.Finish_y)-min(self.Start_y,self.Finish_y) 
            self.box_height = self.box_height*self.height_scale_ratio

            if not (((self.Start_x==self.Finish_x) and
                    (self.Start_y==self.Finish_y)) or
                    ((self.box_X+self.box_width)>self.pixmap_image.width()) or
                    ((self.box_Y+self.box_height)>self.pixmap_image.height())):
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
                self.penRectangle.setWidth(0.5)

                self.painterInstance.setPen(self.penRectangle)
                self.painterInstance.drawRect(self.box_X,self.box_Y,self.box_width,self.box_height)

                self.ui.label.setPixmap(self.pixmap_image)
                self.ui.label.show()
        '''

        self.mouse_x = QMouseEvent.x()
        self.mouse_y = QMouseEvent.y()
        self.update()
        

    def mousePressEvent(self,QMouseEvent):
        self.Start_x = QMouseEvent.x()
        self.Start_y = QMouseEvent.y()
        # self.box_start = True

    def mouseReleaseEvent(self,QMouseEvent):
        # self.box_start = False
        self.Finish_x = QMouseEvent.x()
        self.Finish_y = QMouseEvent.y()
        
        self.box_X = min(self.Start_x,self.Finish_x)-self.ui.label.geometry().x()
        self.box_X = self.box_X*self.width_scale_ratio
        self.box_Y = min(self.Start_y,self.Finish_y)-self.ui.label.geometry().y()
        self.box_Y = self.box_Y*self.height_scale_ratio
        self.box_width = max(self.Start_x,self.Finish_x)-min(self.Start_x,self.Finish_x)
        self.box_width = self.box_width*self.width_scale_ratio
        self.box_height = max(self.Start_y,self.Finish_y)-min(self.Start_y,self.Finish_y) 
        self.box_height = self.box_height*self.height_scale_ratio

        if not (((self.Start_x==self.Finish_x) and
                (self.Start_y==self.Finish_y)) or
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
            self.painterInstance.drawRect(self.box_X,self.box_Y,self.box_width,self.box_height)

            self.ui.label.setPixmap(self.pixmap_image)
            self.ui.label.show()






    @pyqtSlot()
    def open_image(self):
        if self.Init_open:
            self.file_name,_ = QFileDialog.getOpenFileName(self,'Open File', '.')
            self.setWindowTitle(self.file_name)
            self.pixmap_image = QtGui.QPixmap(self.file_name)
             
            self.width_scale_ratio = self.pixmap_image.width()/self.ui.label.geometry().width()
            self.height_scale_ratio = self.pixmap_image.height()/self.ui.label.geometry().height()

            self.painterInstance = QtGui.QPainter(self.pixmap_image)
            
            self.ui.label.setPixmap(self.pixmap_image)
            self.ui.label.show()
            self.Init_open = False

        else:
            del self.painterInstance
            del self.pixmap_image
            
            self.file_name,_ = QFileDialog.getOpenFileName(self,'Open File', '.')
            self.setWindowTitle(self.file_name)
            self.pixmap_image = QtGui.QPixmap(self.file_name)
            
            self.width_scale_ratio = self.pixmap_image.width()/self.ui.label.geometry().width()
            self.height_scale_ratio = self.pixmap_image.height()/self.ui.label.geometry().height()
            
            self.painterInstance = QtGui.QPainter(self.pixmap_image)
            
            self.ui.label.setPixmap(self.pixmap_image)
            self.ui.label.show()
            self.Init_open = False
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    w.show()
    sys.exit(app.exec())

