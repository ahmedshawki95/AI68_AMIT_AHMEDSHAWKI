import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog , QMainWindow,QUndoView,QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox

import cv2
import numpy as np
from PyQt5 import Qt, uic, QtWidgets
from skimage.io import imshow,imread,imsave
class ImageViewer(QMainWindow):
    def __init__(self) -> None:
        """Initialize"""
        # Loading UI form
        super(ImageViewer , self).__init__()
 
        # self.setWindowTitle('Image Viewer')
        uic.loadUi(r'C:\Users\workstation\OneDrive - Alexandria University\AI\computer_vision\day3\task\browse image ,apply median and canny using qt5\task3_gui_v2.ui', self)
        self.layout = QVBoxLayout()

        # Find the QLabel from the UI (assuming it already exists in the .ui file)
        self.originview = self.findChild(QLabel, 'origin')


        #self.line_edit = QLineEdit(self)

        self.line_edit = self.findChild(QLineEdit, 'lineEdit')
        self.line_edit_2 = self.findChild(QLineEdit, 'lineEdit_2')



        self.browse_image.clicked.connect(self.openFileDialog)
        self.apply_filter.clicked.connect(self.apply_filter_fn)
        self.detect_edges.clicked.connect(self.apply_canny_fn)


        self.img = None
    
    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if fileName:
            self.img = cv2.imread(fileName)
            self.displayImage(self.img, self.originview)
    
    def displayImage(self, img, label):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage into the QLabel as a QPixmap and scale it to fit the QLabel's size
        label.setPixmap(QPixmap.fromImage(qImg).scaled(label.size(), aspectRatioMode=1))

    def apply_filter_fn(self):

        try:
            self.img = cv2.medianBlur(self.img, 5)
            self.displayImage(self.img, self.after_filter_label)
        except Exception as e:
            self.show_error_dialog("Error", "upload image first.")

        

    def apply_canny_fn(self):
        try:
            threshold1 = int(self.line_edit.text())
            threshold2 = int(self.line_edit_2.text())
            self.img_canny = cv2.Canny(self.img, threshold1,threshold2)
            self.view_after_canny_fn(self.img_canny, self.after_edge_detection_label)
        except Exception as e:
            self.show_error_dialog("Error", "upload image first.")


    def view_after_canny_fn(self, img, label):
        height, width = img.shape
        bytesPerLine = width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

        # Set the QImage into the QLabel as a QPixmap and scale it to fit the QLabel's size
        label.setPixmap(QPixmap.fromImage(qImg).scaled(label.size(), aspectRatioMode=1))

    def show_error_dialog(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.exec_()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
