import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, \
    QDesktopWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtCore

from src.UI.zoomWindow import ZoomWindow, ZoomableImage
from src.model.landmarkcaliper import LandmarkCaliper

class LaunchApp(QMainWindow):
    image_file_path = ""
    processed_image_path = ""
    output_files = []
    zoomWindow = ""

    def __init__(self):
        super().__init__()

        uic.loadUi("Layout2.ui", self)

        self.imageViewer.setScaledContents(True)
        self.setMaximumHeight(self.get_max_window_height())
        self.setMaximumWidth(self.get_max_window_width())

        self.open_file_dialog_button = self.findChild(QPushButton, "uploadButton")
        self.run_detection_button = self.findChild(QPushButton, "submitButton")
        self.open_new_button = self.findChild(QPushButton, "openNewButton")

        self.open_file_dialog_button.clicked.connect(self.open_file_dialog)
        self.run_detection_button.clicked.connect(self.run_detection)
        self.showLandmarksButton.clicked.connect(self.showLandmarksImage)
        self.showDetailsButton.clicked.connect(self.showDetailedImage)
        self.open_new_button.clicked.connect(self.openNewWindow)

        self.show()

    def open_file_dialog(self):
        self.result.clear()
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, "Select an image of a hand", "", "Image Files (*.png *.jpg *.jpeg)")[0]
        if file_path:
            print(f"Selected file: {file_path}")
            self.image_file_path = file_path
            self.output_files = [] #clear old detection results
            self.display_image(file_path)

    def run_model(self) -> list[str]:
        caliper = LandmarkCaliper()
        caliper.measure(self.image_file_path)
        return caliper.get_measurement_files()


    def display_image(self, file_path):
        self.processed_image_path = file_path
        pixmap = QPixmap(file_path)
        self.imageViewer.setPixmap(pixmap.scaled(self.imageViewer.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    @staticmethod
    def get_max_window_height():
        screen = QDesktopWidget().screenGeometry()
        max_height = screen.height() * 0.9  # Set the maximum height to 80% of the screen height
        return int(max_height)

    @staticmethod
    def get_max_window_width():
        screen = QDesktopWidget().screenGeometry()
        max_width = screen.width() * 0.9  # Set the maximum height to 80% of the screen height
        return int(max_width)

    def run_detection(self):
        if len(self.image_file_path) == 0:
            self.showMessageBox("Please upload an image of a hand first")
        else:
            patient_id = self.patientId.toPlainText()
            patient_name = self.patientName.toPlainText()
            self.output_files = self.run_model()
            if len(self.output_files) != 0:
                landmark_image = self.output_files[4]
                self.display_image(landmark_image)
                self.showLandmarksButton.checked = True
                self.result.textCursor().insertHtml(f'Detection finished for {patient_name} with Id: {patient_id}.<br><br>'
                                     f'Results saved in following files:<br>'
                                     f'<b>Identified Landmarks Image</b>: {self.output_files[4]}<br><br>'
                                     f'<b>Dimension Info Image</b>: {self.output_files[2]}<br><br>'
                                     f'<b>Selected Landmarks Dimensions CSV</b>: {self.output_files[0]}<br><br>'
                                     f'<b>All Landmarks Dimensions CSV</b>: {self.output_files[1]}<br>')
            else:
                self.showMessageBox("No landmarks detected. Please make sure the uploaded image is of the back of a "
                                    "hand.")

    def showLandmarksImage(self):
        if len(self.output_files) < 5:
            self.showMessageBox("Processed image is not ready. Please make sure detection is executed first!")
        else:
            landmark_image = self.output_files[4]
            self.display_image(landmark_image)

    def showDetailedImage(self):
        if len(self.output_files) < 3:
            self.showMessageBox("Processed image is not ready. Please make sure detection is executed first!")
        else:
            detail_image = self.output_files[2]
            self.display_image(detail_image)

    def openNewWindow(self):
        if len(self.processed_image_path) == 0:
            self.showMessageBox("Please upload and image first")
        else:
            self.zoomWindow = ZoomableImage(self.processed_image_path, self)
            self.zoomWindow.show()


    def showMessageBox(self, text):
        msg = QMessageBox()
        msg.setWindowTitle("Landmark Caliper")
        msg.setText(text)
        msg.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LaunchApp()
    # window.show()
    sys.exit(app.exec_())
