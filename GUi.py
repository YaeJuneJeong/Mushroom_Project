import sys
from threading import Thread
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, QRect, QPoint, QPropertyAnimation, QThread, pyqtSignal, QWaitCondition
from PyQt5.QtWidgets import QMessageBox

import requests

url = "http://54.210.105.132/api/image/upload"
title = 'title'
file_path = 'D:/example1.jpg'


class Send(QThread):
    emit = pyqtSignal(int)

    def __init__(self):
        QThread.__init__(self)
        self.cond = QWaitCondition()
        self._status = False

    def run(self):
        try:
            data = [('mushroomId', 17)]
            response = requests.post(url, data=data, timeout=10)

            if response.text is not 201:
                print(response.text)
                raise Exception

        except TimeoutError:
            QMessageBox.about(self, 'Error', "서버 에러입니다 네트워크 연결을 다시 확인해 주세요")
        except Exception:
            QMessageBox.about(self, 'Error', "핀번호를 다시 확인해 주세요")
        finally:
            self.emit.emit(1)


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
            th1 = Thread(target=self.send)
            th1.start()

            self.state = False

    def on_valueChanged(self, value):
        t = QtGui.QTransform()
        t.rotate(value)
        self.setPixmap(self._pixmap.transformed(t))

    def send(self):
        try:
            data = [('mushroomId', 17)]
            response = requests.post(url, data=data, timeout=10)

            if response.text is not 201:
                print(response.text)
                raise Exception

        except TimeoutError:
            QMessageBox.about(self, 'Error', "서버 에러입니다 네트워크 연결을 다시 확인해 주세요")
        except Exception:
            QMessageBox.about(self, 'Error', "핀번호를 다시 확인해 주세요")
        finally:
            self.sendEvent.emit(1)


class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)
        self.th = QThread(self)
        self.send = Send()
        self.send.moveToThread(self.th)

        label = RotateMe(alignment=QtCore.Qt.AlignCenter)
        label.set_pixmap(QtGui.QPixmap('./operating.png'))
        button = QtWidgets.QPushButton('Rotate')
        button.clicked.connect(label.start_animation)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(label)
        lay.addWidget(button)




if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
