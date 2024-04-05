import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt


class ZoomableImage(QWidget):

    image_path = ""

    def __init__(self, image_path, parent=None):
        print("reached ZoomableImage begin")
        #super().__init__(parent)
        super().__init__()

        self.setWindowTitle("Zoomable View")

        self.setMinimumHeight(900)
        self.setMinimumWidth(880)

        self.image_label = QLabel(self)
        self.pixmap = QPixmap(image_path)
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

        self.zoom_level = 0

        self.zoom_in_button = QPushButton("Zoom In", self)
        self.zoom_in_button.setStyleSheet("background-color: qlineargradient(spread:reflect, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(47, 159, 185, 255), stop:0.625164 rgba(202, 232, 255, 255));")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out", self)
        self.zoom_out_button.setStyleSheet(
            "background-color: qlineargradient(spread:reflect, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(47, 159, 185, 255), stop:0.625164 rgba(202, 232, 255, 255));")

        self.zoom_out_button.clicked.connect(self.zoom_out)
        layout.addWidget(self.zoom_out_button)
        print("reached ZoomableImage end")

    def zoom_in(self):
        # Zoom in
        if self.zoom_level < 10:  # Limit maximum zoom level
            self.zoom_level += 1
            self.update_image_size()

    def zoom_out(self):
        # Zoom out
        if self.zoom_level > -10:  # Limit minimum zoom level
            self.zoom_level -= 1
            self.update_image_size()

    def update_image_size(self):
        scale_factor = 1.25 ** self.zoom_level
        new_size = scale_factor * self.pixmap.size()
        self.image_label.setPixmap(self.pixmap.scaled(new_size))

    def set_image_path(self, path):
        self.image_path = path

class ZoomWindow(QWidget):
    def __init__(self, image_path):
        print("reached ZoomWindow begin")
        super().__init__()
        self.setWindowTitle("Zoomable View")
        self.zoomable_image = ZoomableImage(image_path, self)
        print("reached ZoomWindow end")




