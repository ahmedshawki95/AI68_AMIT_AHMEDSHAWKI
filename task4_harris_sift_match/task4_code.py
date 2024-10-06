import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog ,QWidget
from PyQt5.QtGui import QPixmap, QImage ,QPainter
from PyQt5.QtWidgets import QMessageBox,QSlider, QMainWindow,QUndoView,QLineEdit,QGraphicsView,QGraphicsScene
import cv2
import numpy as np
from PyQt5 import Qt, uic, QtWidgets
from skimage.io import imshow,imread,imsave
from PyQt5.QtCore import Qt  # Import the correct Qt module
from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QMovie
import requests  # Add this line




class ImageViewer(QMainWindow):
    def __init__(self) -> None:
        """Initialize"""
        # Loading UI form
        super(ImageViewer , self).__init__()
 
        # self.setWindowTitle('Image Viewer')
        uic.loadUi(r'C:\Users\workstation\OneDrive - Alexandria University\AI\computer_vision\day4\task4_harris_sift_match\task4_gui.ui', self)
        self.layout = QVBoxLayout()

        self.browse_image.clicked.connect(lambda: self.openFileDialog(1))
        self.apply_harris.clicked.connect(self.apply_harris_fn)
        self.apply_sift.clicked.connect(self.apply_sift_fn)
        self.upload_2nd_img.clicked.connect(lambda: self.openFileDialog(2))
        self.apply_match.clicked.connect(self.matching)

        self.update_label()
        self.horizontalSlider_harris_threshold.valueChanged.connect(self.update_label)

        self.img = None
    
    def openFileDialog(self,n):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        if fileName:
            if n==1:
                self.img = cv2.imread(fileName)
                self.displayImage(self.img, self.original_graphicsView)
            elif n==2:
                self.img2 = cv2.imread(fileName)
                self.displayImage(self.img2, self.graphicsView_2nd_img)

    def displayImage(self, img, graphicsView):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qImg)

        # Scale the QPixmap to the size of the QGraphicsView, keeping aspect ratio
        scaledPixmap = pixmap.scaled(graphicsView.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create a QGraphicsScene and add the scaled pixmap to it
        scene = QGraphicsScene()
        scene.addPixmap(scaledPixmap)

        # Set the scene to the QGraphicsView
        graphicsView.setScene(scene)

        # Ensure the image scales dynamically when the view is resized
        graphicsView.setRenderHint(QPainter.SmoothPixmapTransform)
        graphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)




    def apply_harris_fn(self):

        try:
            self.img_after_harris = self.img.copy()
            self.gray = cv2.cvtColor(self.img_after_harris,cv2.COLOR_BGR2GRAY)
            self.gray = np.float32(self.gray)
            self.corner_img = cv2.cornerHarris(self.gray,2,3,0.04)
            self.dilate_img = cv2.dilate(self.corner_img,None)
            self.img_after_harris[self.dilate_img>(self.harris_value*self.dilate_img.max())]=[0,0,255]

            self.displayImage(self.img_after_harris, self.harris_graphicsView)
        except Exception as e:
            self.show_error_dialog("Error", "upload image first.")


    def apply_sift_fn(self):
        try:
            self.img_after_sift = self.img.copy()
            self.gray= cv2.cvtColor(self.img_after_sift,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp = sift.detect(self.gray,None)
            self.img_after_sift=cv2.drawKeypoints(self.gray,kp,self.img_after_sift)
            
            self.displayImage(self.img_after_sift, self.sift_graphicsView)

        except Exception as e:
            self.show_error_dialog("Error", "upload image first.")
            
    def matching(self):
        try:
            # Create SIFT detector
            sift = cv2.SIFT_create()
            # Detect keypoints and compute descriptors for both images
            kp1, des1 = sift.detectAndCompute(self.img, None)
            kp2, des2 = sift.detectAndCompute(self.img2, None)
            # Create a Brute Force Matcher object
            bf = cv2.BFMatcher()
            # Match descriptors
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            # Draw the matches
            self.matching_result = cv2.drawMatches(self.img, kp1, self.img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.displayImage(self.matching_result, self.graphicsView_matching)

            if len(good_matches) > 10:
                print("match")
                self.match_result.setText(f"images does match")
            else:
                print("not match")
                self.match_result.setText(f"images does not match")

        except Exception as e:
            self.show_error_dialog("Error", "upload images first.")


    def show_error_dialog(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.exec_()

    def update_label(self):
        # Get the current value of the slider
        self.harris_value = self.horizontalSlider_harris_threshold.value()/100
        # Update the label text with the slider's value
        self.harris_threshold.setText(f"Harris Threshold: {self.harris_value}")    






if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
