import base64
import datetime
import json
import threading
import time

import serial
import socketio
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import requests
import socket
import cv2
import pyrealsense2 as rs
import numpy as np

url_register = "http://184.73.45.24/api/myfarm/register/ip"
url_info = "http://184.73.45.24/api/myfarm/info"

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
local_ip_address = s.getsockname()[0]

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
user_id = 0
ret = 0

# socket 서버 연결
sio = socketio.Client()
sio.connect('http://localhost:3001')


@sio.on("req_cosdata")
def socket_data(temp, hum):
    sio.emit("res_cosdata", {"temperature": temp, "humidity": hum})


@sio.on("req_video")
def socket_stream(req):
    if req:
        stream_end_value = True
    t = threading.Thread(target=stream_thread)


@sio.on("req_video")
def stream_thread(req):
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        result, frame = cv2.imencode('.jpg', color_image, encode_param)
        data = base64.b64encode(frame)

        sio.emit('req_image', data)
        if req:
            break


SERVER_URL = 'http://184.73.45.24/api'  # 서버 url
PIN = '107512'  # 기기 고유 핀번호

R = "R"  # 환경 데이터 프로토콜
C = "C"  # 온도 프로토콜
S = "S"  # 습도 프로토콜
WATER_ORDER = 'W'  # 물주기 프로토콜
MOTER_ORDER = 'M'  # 3D 촬영 프로토콜

HOUR = 100  # 온도 데이터 전송 시간 기준값
WATER_TIME = 0  # 물주는 시간 기준값
MOTER_TIME = HOUR * 6  # 모터 시간 기준값
D2_TIME = HOUR

water_num = 0

data = None  # 환경 데이터 전역

url = SERVER_URL + "/image/upload"
title = 'title'
file_path = 'D:/example1.jpg'


# 시작전 polling 코드
class Start_before(QThread):
    errors = pyqtSignal(int)
    finished = pyqtSignal()
    global data
    global water_num

    id_value = ''

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False

    def run(self):
        try:
            while True:
                time.sleep(1)  # 429 에러 방지
                params = {'pin': PIN}
                response = requests.get(
                    SERVER_URL + '/farm/exist',
                    params=params
                )
                print(response.status_code)
                if response.status_code == 200:
                    id_value = response.text
                    break

            while self.isRun:
                params = {'id': id_value, 'type': 'custom'}
                response = requests.get(SERVER_URL + '/farm/data', params=params)
                result = json.loads(response.text)
                water_num = result['water']
                temp_array = result['temperature']
                hum_array = result['humidity']

                data_len = len(temp_array)

                data = [None] * data_len

                for i in range(0, data_len):
                    temp_array[i] = str(temp_array[i]['setting_value'])
                    hum_array[i] = str(hum_array[i]['setting_value'])
                    data[i] = R + C + temp_array[i] + S + hum_array[i]

                print(f"물 횟수 : {water_num}")
                print(f"온도 : {temp_array}")
                print(f"습도 : {hum_array}")
                print(f"데이터 : {data}")

                self.finished.emit(100)
                self.isRun = False
                break
        except Exception:
            self.errors.emit(100)
            self.isRun = False


def refresh_data():
    refresh_url = SERVER_URL + '/api/myfarm/data/hour'
    data = {'id': 2, 'ip': local_ip_address}
    response = requests.put(refresh_url, data=data, timeout=10)


def encode_serial_data(str):
    return str.encode('utf-8')


# 기기 가동 코드
class Start(QThread):
    error = pyqtSignal(int)
    finished = pyqtSignal()
    global water_num
    global data
    pipeline = None
    global D2_TIME

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False
        self.temp = None
        self.hum = None

    def run(self):
        try:

            WATER_TIME = HOUR * 24 / water_num

            serial_send_len = 0
            hour = 0  # 시간 초로 변환
            water_time = WATER_TIME  # 값 받아 오면 연산할 물주기 시간
            moter_time = MOTER_TIME

            now = datetime.datetime.now()
            Arduino = serial.Serial(port='COM7', baudrate=9600)
            # self.isRun = True

            while self.isRun:
                dt1 = datetime.datetime.now()
                result = dt1 - now
                seconds = int(result.total_seconds())
                print(1)
                if Arduino.readable():
                    LINE = Arduino.readline()
                    code = str(LINE.decode().replace('\n', ''))
                    print(code)
                    hum = code[10: 12]
                    temp = code[30: 32]
                    socket_data(hum, temp)
                    self.hum = hum
                    self.temp = temp

                if seconds == hour:
                    Arduino.write(encode_serial_data(data[serial_send_len]))

                    serial_send_len += 1
                    hour += HOUR

                if seconds == water_time:
                    Arduino.write(encode_serial_data(WATER_ORDER))
                    water_time += WATER_TIME

                if seconds == moter_time:
                    Arduino.write(encode_serial_data(MOTER_ORDER))
                    moter_time += MOTER_TIME

                if seconds == D2_TIME:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()

                    # Convert images to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.imwrite('./color_sensor.jpg', color_image)

                    files = {'mushroom': ('mushroom12.jpg', open('./color_sensor.jpg', 'rb'))}
                    data = [('mushroomId', 17)]
                    response = requests.post(url, files=files, data=data)
                    print(response.status_code)
                    D2_TIME += 1000
                # 끝내기 데이터 오면 break 후 리턴
                self.finished.emit()
        except Exception:
            self.error.emit(100)
            self.isRun = False


class Send(QThread):
    error = pyqtSignal(int)
    info = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False
        global ret

    def run(self):
        try:
            while self.isRun:
                # change ip address
                data = {'id': 2, 'ip': local_ip_address}
                response = requests.put(url_register, data=data, timeout=10)
                print(response.status_code)

                if response.status_code != 200:
                    raise Exception

                # response = requests.get(url_info, params={'id': 2})
                # print(response.json())
                # if (response.json()['machine_userid']) is not 2:
                #     self.info.emit(response.json())
                #     break
                # time.sleep(10)

                # user_id
                # response.json()['machine_userid']

                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asarray(color_frame.get_data())
                cv2.imwrite('./recent.jpg', color_image)
                self.isRun = False
                self.finished.emit()
                break
        except Exception:
            self.error.emit(100)
            self.isRun = False


# class RotateMe(QtWidgets.QLabel, QThread):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._pixmap = QtGui.QPixmap()
#         self.state = False
#         self._animation = QtCore.QVariantAnimation(
#             self,
#             startValue=0.0,
#             endValue=360.0,
#             duration=1000,
#             valueChanged=self.on_valueChanged
#         )
#         self._animation.setLoopCount(-1)
#
#     def set_pixmap(self, pixmap):
#         self._pixmap = pixmap
#         self.setPixmap(self._pixmap)
#
#     def start_animation(self):
#         if self.state is False:
#             self._animation.start()
#             self.state = True
#
#     def stop_animations(self):
#         if self.state is True:
#             self._animation.stop()
#             self.state = False
#
#     def on_valueChanged(self, value):
#         t = QtGui.QTransform()
#         t.rotate(value)
#         self.setPixmap(self._pixmap.transformed(t))


class Window1(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Window1, self).__init__(parent)

        self.secondWindow = Window2()
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.movie = QtGui.QMovie('./giphy.gif')
        self.label.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()
        self.send = Send(self)
        self.send.finished.connect(self.stop)
        self.send.finished.connect(self.change_stack)
        # self.send.info.connect(self.get_info)
        self.send.error.connect(self.error)

        button = QtWidgets.QPushButton('ip 저장')
        # button.clicked.connect(self.label.start_animation)
        button.clicked.connect(self.start)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(button)

    def start(self):
        if not self.send.isRun:
            self.send.isRun = True
            self.send.start()
            self.movie.start()

    def stop(self):
        if self.send.isRun:
            self.send.isRun = False
            self.movie.stop()

    def error(self):
        self.send.isRun = False
        self.movie.stop()
        QMessageBox.information(self, 'Error', 'Please check Network')

    def change_stack(self):
        self.parent().stack.setCurrentIndex(1)

    def get_info(self):
        self.parent().change(data)


class Window2(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Window2, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout(self)
        button = QtWidgets.QPushButton('배지 확인')
        button.clicked.connect(self.check)
        self.layout.addWidget(button)

    # def set(self, data):
    # print(data)
    # label = QtWidgets.QLabel(str(data['id']))
    # self.layout.addWidget(label)
    #
    # label = QtWidgets.QLabel(str(data['machine_ip']))
    # self.layout.addWidget(label)
    #
    # label = QtWidgets.QLabel(str(data['machine_name']))
    # self.layout.addWidget(label)

    def check(self):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data())

        # take the only location of mushroom pot -> 1/3 * width,1/2*height
        recent_image = color_image[240:, 214:428]
        check_image = cv2.imread('./recent.jpg')[240:, 214:428]

        cv2.imwrite('./check.jpg', color_image)
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
            response = requests.get(SERVER_URL + 'myfarm/status', params={'id': 2, 'status': 'true'})
            self.change_stack()

    def change_stack(self):
        self.parent().stack.setCurrentIndex(2)


class Window3(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Window3, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel('그래프 화면 입니다')
        self.layout.addWidget(label)

        self.start_before = Start_before(self)
        self.start = Start(self)

        self.start_before.finished.connect(self.go)
        self.start.finished.connect(self.renewal)
        self.data1 = QtWidgets.QLabel(str(self.start.hum))
        self.data2 = QtWidgets.QLabel(str(self.start.temp))

        self.layout.addWidget(self.data1)
        self.layout.addWidget(self.data2)
        self.before_run()

    def before_run(self):
        if not self.start_before.isRun:
            self.start_before.isRun = True
            self.start_before.start()

    def go(self):
        if not self.start.isRun:
            self.start.start()

    def renewal(self):
        self.data1 = QtWidgets.QLabel(str(self.start.hum))
        self.data2 = QtWidgets.QLabel(str(self.start.temp))
        self.layout.addWidget(self.data1)
        self.layout.addWidget(self.data2)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("버섯 GUI")
        self.setWindowIcon(QtGui.QIcon('mushroom.jpg'))
        self.stack = QtWidgets.QStackedLayout(self)
        self.stack1 = Window1(self)
        self.stack2 = Window2(self)
        self.stack3 = Window3(self)
        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.stack.addWidget(self.stack3)
        self.show()

    def change(self, data):
        self.stack2.set(data)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main = MainWindow()
    # main = Window1()
    app.exit(app.exec_())
