import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QMainWindow
import requests
import socket
import cv2
import pyrealsense2 as rs
import numpy as np

url_register = "http://54.210.105.132/api/myfarm/register/ip"
url_info = "http://54.210.105.132/api/myfarm/info"

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
local_ip_address = s.getsockname()[0]

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

user_id = 0

class Send(QThread):
    error = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False

    def run(self):
        try:
            while self.isRun:
                # change ip address
                data = {'id': 14, 'ip': '141414441311.12'}
                response = requests.put(url_register, data=data, timeout=10)
                print(response.status_code)

                if response.status_code != 200:
                    raise Exception
                while True:

                    response = requests.get(url_info, params={'id': 14})
                    if(response.json()['machine_userid']) is not 1:
                        break
                    time.sleep(10)

                # user_id
                # response.json()['machine_userid']

                self.finished.emit()
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asarray(color_frame.get_data())
                cv2.imwrite('./recent.jpg', color_image)
                self.isRun = False
                break
        except Exception:
            self.error.emit(100)
            self.isRun = False


class RotateMe(QtWidgets.QLabel, QThread):
    def __init__(self, *args, **kwargs):
        super(RotateMe, self).__init__(*args, **kwargs)
        self._pixmap = QtGui.QPixmap()
        self.state = False
        self._animation = QtCore.QVariantAnimation(
            self,
            startValue=0.0,
            endValue=360.0,
            duration=1000,
            valueChanged=self.on_valueChanged
        )
        self._animation.setLoopCount(-1)

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.setPixmap(self._pixmap)

    def start_animation(self):
        if self.state is False:
            self._animation.start()
            self.state = True

    def stop_animations(self):
        if self.state is True:
            self._animation.stop()
            self.state = False

    def on_valueChanged(self, value):
        t = QtGui.QTransform()
        t.rotate(value)
        self.setPixmap(self._pixmap.transformed(t))


class Window1(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Window1, self).__init__(parent)

        self.label = RotateMe(alignment=QtCore.Qt.AlignCenter)
        self.label.set_pixmap(QtGui.QPixmap('./operating.png'))

        self.send = Send(self)
        self.send.finished.connect(self.stop)
        self.send.finished.connect(self.change_stack)
        self.send.error.connect(self.error)

        button = QtWidgets.QPushButton('Submit')
        button.clicked.connect(self.label.start_animation)
        button.clicked.connect(self.start)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(button)


    def start(self):
        if not self.send.isRun:
            self.send.isRun = True
            self.send.start()

    def stop(self):
        if self.send.isRun:
            self.send.isRun = False
            self.label.stop_animations()

    def error(self):
        self.send.isRun = False
        self.label.stop_animations()
        QMessageBox.information(self, 'Error', 'Please check Network')

    def change_stack(self):
        self.parent().stack.setCurrentIndex(1)


class Window2(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Window2, self).__init__(parent)

        button = QtWidgets.QPushButton('배지 확인')
        button.clicked.connect(self.check)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(button)

    def check(self):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data())

        # take the only location of mushroom pot -> 1/3 * width,1/2*height
        recent_image = color_image[240:, 214:428]
        check_image = cv2.imread('./recent.jpg')[240:, 214:428]

        cv2.imwrite('./check.jpg',color_image)
        hist_recent = cv2.calcHist(recent_image, [1], None, [255], [0, 255])
        hist_check = cv2.calcHist(check_image, [1], None, [255], [0, 255])
        number = cv2.compareHist(hist_recent, hist_check, cv2.HISTCMP_CORREL)

        print(number)
        if number > 0.4:
            QMessageBox.information(self, 'Error', '배지를 넣어 주세요')
        else:
            #            배지입력 확인
            #
            print('Success')
            self.change_stack()

    def change_stack(self):
        self.parent().stack.setCurrentIndex(0)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.stack = QtWidgets.QStackedLayout(self)
        self.stack1 = Window1(self)
        self.stack2 = Window2(self)
        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main = MainWindow()
    app.exit(app.exec_())
