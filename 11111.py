import requests
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QSize
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

SERVER_URL = 'http://184.73.45.24/api'  # 서버 url
PIN = '107512'  # 기기 고유 핀번호

def drawline(img, pt1, pt2, color, thickness=3, style='dotted', gap=30):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = False

    def run(self):
        # capture from web cam
        profile = pipeline.start()
        self._run_flag = True
        while self._run_flag:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            img = np.asarray(color_frame.get_data())
            self.change_pixmap_signal.emit(img)
        # shut down capture system
        pipeline.stop()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data())

        # take the only location of mushroom pot -> 1/3 * width,1/2*height
        recent_image = color_image[240:, 214:428]
        check_image = cv2.imread('./recent.jpg')[240:, 214:428]

        hist_recent = cv2.calcHist(recent_image, [1], None, [255], [0, 255])
        hist_check = cv2.calcHist(check_image, [1], None, [255], [0, 255])
        number = cv2.compareHist(hist_recent, hist_check, cv2.HISTCMP_CORREL)

        # print(number)
        # if number > 0.4:
        #     error_dialog = QtWidgets.QErrorMessage()
        #     error_dialog.showMessage('Oh no!')
        # else:
        #     #            배지입력 확인
        #     #
        #     print('Success')
        #     response = requests.get(SERVER_URL + 'myfarm/status', params={'id': 2, 'status': 'true'})
        # pipeline.stop()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.scale = 1
        # self.setWindowTitle("Qt live label demo")
        self.display_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(320, 240)
        #

        self.temp_2 = QLabel(self)
        self.temp_2.resize(QtGui.QPixmap('./logo1.png').width(), QtGui.QPixmap('./logo1.png').height())
        self.temp_2.setPixmap(QtGui.QPixmap('./logo1.png'))

        self.temp_3 = QLabel(self)
        self.temp_3.resize(QtGui.QPixmap('./logo.png').width(),QtGui.QPixmap('./logo.png').height())
        logo = QtGui.QPixmap('./logo.png')
        self.scale /= 9
        logo = logo.scaled(logo.size() * self.scale)
        self.temp_3.setPixmap(logo)


        # create a text label
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setText('キット確認')
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton{color: white;"
                                      "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, "
                                      "stop:0 rgba(255, 190, 11, 255), stop:1 rgba(251, 86, 7, 255));"
                                      "border-radius:20px}")
        self.pushButton.setFixedSize(150,100)
        self.pushButton.clicked.connect(self.judge)
        # self.pushButton.setGeometry(QtCore.Qt.AlignCenter)
        hbox1 = QHBoxLayout()
        hbox1.addStretch(2)
        hbox1.addWidget(self.pushButton)
        hbox1.addStretch(2)

        hbox2 = QHBoxLayout()
        hbox2.addStretch(1)
        hbox2.addWidget(self.image_label)
        hbox2.addStretch(1)

        hbox3 = QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addWidget(self.temp_3)
        hbox3.addWidget(self.temp_2)
        hbox3.addStretch(1)
        # create a vertical box layout and add the two labels

        hbox4  = QHBoxLayout()
        title = QLabel('キットの位置を点線にマッチングしてください')
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        hbox4.addStretch(1)
        hbox4.addWidget(title)
        hbox4.addStretch(1)
        #
        vbox = QVBoxLayout()
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox1)
        # self.pushButton.setFixedSize(QSize(100,100))
        # vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()

    def judge(self):

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data())

        # take the only location of mushroom pot -> 1/3 * width,1/2*height
        recent_image = color_image[240:, 214:428]
        check_image = cv2.imread('./recent.jpg')[240:, 214:428]

        hist_recent = cv2.calcHist(recent_image, [1], None, [255], [0, 255])
        hist_check = cv2.calcHist(check_image, [1], None, [255], [0, 255])
        number = cv2.compareHist(hist_recent, hist_check, cv2.HISTCMP_CORREL)
        print(number)
        if number > 0.4:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Oh no!')
        else:
            #            배지입력 확인
            #
            print('Success')
            response = requests.get(SERVER_URL + 'myfarm/status', params={'id': 2, 'status': 'true'})
        pipeline.stop()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        # print(h,w,ch)
        bytes_per_line = ch * w
        drawrect(rgb_image, (320, 122), (960, 700), (255, 124, 2), 4, 'dotted')
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width/2, self.display_height/2, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())