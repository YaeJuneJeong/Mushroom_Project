from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import requests
import socket

url = "http://54.210.105.132/api/myfarm/register/ip"
title = 'title'
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
local_ip_address = s.getsockname()[0]


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
                data = {'id': 14, 'ip': '123.1212.2231.1'}
                response = requests.put(url, data=data, timeout=10)

                print(response.status_code)

                if response.status_code != 200:
                    raise Exception
                self.finished.emit()
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


class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        self.label = RotateMe(alignment=QtCore.Qt.AlignCenter)
        self.label.set_pixmap(QtGui.QPixmap('./operating.png'))
        self.show()
        self.send = Send(self)
        self.send.finished.connect(self.stop)
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


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    # w.show()
    sys.exit(app.exec_())