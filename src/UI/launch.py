import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, \
    QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5 import uic, QtCore

from src.model.landmarkcaliper import LandmarkCaliper

class LaunchApp(QMainWindow):
    image_file_path = ""

    def __init__(self):
        super().__init__()

        uic.loadUi("Layout.ui", self)

        self.imageViewer.setScaledContents(True)
        self.setMaximumHeight(self.get_max_window_height())
        self.setMaximumWidth(self.get_max_window_width())

        self.open_file_dialog_button = self.findChild(QPushButton, "uploadButton")
        self.run_detection_button = self.findChild(QPushButton, "submitButton")

        self.open_file_dialog_button.clicked.connect(self.open_file_dialog)
        self.run_detection_button.clicked.connect(self.run_detection)
        self.show()

    def open_file_dialog(self):
        self.message.setText("")
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")[0]
        if file_path:
            print(f"Selected file: {file_path}")
            self.image_file_path = file_path
            self.display_image(file_path)

    def run_model(self) -> list[str]:
        caliper = LandmarkCaliper()
        caliper.measure(self.image_file_path)
        return caliper.get_measurement_files()

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.imageViewer.setPixmap(pixmap.scaled(self.imageViewer.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def get_max_window_height(self):
        screen = QDesktopWidget().screenGeometry()
        max_height = screen.height() * 0.9  # Set the maximum height to 80% of the screen height
        return int(max_height)

    def get_max_window_width(self):
        screen = QDesktopWidget().screenGeometry()
        max_width = screen.width() * 0.9  # Set the maximum height to 80% of the screen height
        return int(max_width)

    def run_detection(self):
        patientId = self.patientId.toPlainText()
        patientName = self.patientName.toPlainText()
        output_files = self.run_model()
        landmark_image = output_files[2]
        self.display_image(landmark_image)
        self.message.setText(f'Detection finished for {patientName}. \n\n Image with landmarks with measurements saved at : {landmark_image} \n\n File with measurements saved at: {output_files[1]}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LaunchApp()
    # window.show()
    sys.exit(app.exec_())
